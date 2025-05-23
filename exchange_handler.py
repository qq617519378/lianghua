# exchange_handler.py
import asyncio
import base64
import hashlib
import hmac
import json
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, Optional, Callable, Awaitable, List

import httpx
import self
import websockets
from loguru import logger  # 使用 loguru 替换标准 logging，因其在异步场景下更方便
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError
from urllib.parse import urlencode

# --- 1. 抽象基类 (exchange/base.py) ---
class OrderResult(BaseModel):
    ok: bool
    order_id: Optional[str] = None  # 交易所返回的订单ID
    client_order_id: Optional[str] = None  # 客户端自定义订单ID (如果API支持)
    data: Dict[str, Any] = Field(default_factory=dict)  # <--- 修改点：移除了 Optional，类型为 Dict
                                                      # 原始API响应数据，默认为新的空字典
    error_code: Optional[str] = None
    error_message: Optional[str] = None


class BaseExchangeClient(ABC):
    """交易所接口统一抽象"""
    account_name: str  # 用于日志和区分

    @abstractmethod
    async def get_balance(self, currency: str) -> Dict[str, Any]:
        """获取特定币种余额, 返回包含 'total', 'available', 'frozen' 的字典"""
        ...

    @abstractmethod
    async def get_all_balances(self) -> Dict[str, Dict[str, Any]]:
        """获取所有非零币种余额"""
        ...

    @abstractmethod
    async def place_order(
            self,
            inst_id: str,  # 产品ID，如 BTC-USDT-SWAP
            side: str,  # buy 或 sell
            ord_type: str,  # market, limit, post_only, fok, ioc
            size: str,  # 下单数量，对于币币/币币杠杆市价单，基准币种为买单，计价币种为卖单；对于其他产品类型，皆为张
            px: Optional[str] = None,  # 委托价格，仅适用于 limit, post_only, fok, ioc 类型的订单
            td_mode: str = "cross",  # 交易模式: isolated：逐仓 cross：全仓 cash：非保证金
            cl_ord_id: Optional[str] = None,  # 客户端自定义订单ID
            **kwargs
    ) -> OrderResult: ...

    @abstractmethod
    async def cancel_order(self, inst_id: str, ord_id: Optional[str] = None,
                           cl_ord_id: Optional[str] = None) -> OrderResult: ...

    @abstractmethod
    async def get_order_details(self, inst_id: str, ord_id: Optional[str] = None,
                                cl_ord_id: Optional[str] = None) -> OrderResult: ...

    @abstractmethod
    async def get_positions(self, inst_id: Optional[str] = None, inst_type: str = "SWAP") -> List[Dict[str, Any]]: ...

    @abstractmethod
    async def subscribe_private_data(self, ws_event_callback: Callable[[str, Dict[str, Any]], Awaitable[None]]) -> None:
        """启动并维持私有 WebSocket（订单/持仓/账户余额推送）"""
        ...

    @abstractmethod
    async def close_connection(self) -> None:
        """关闭所有网络连接 (HTTP客户端和WebSocket)"""
        ...


# --- 2. OKX REST (exchange/okx_rest.py) ---
OKX_REST_URL = "https://www.okx.com"
_SIM_HEADER = {"x-simulated-trading": "1"}


class OKXRestClient(BaseExchangeClient):
    def __init__(
            self,
            account_name: str,
            api_key: str,
            api_secret: str,
            passphrase: str,
            demo_mode: bool = True,  # 从全局配置获取
            *,
            timeout: float = 10.0,
    ):
        self.account_name = account_name
        self.key = api_key
        self.secret = api_secret.encode('utf-8')
        self.passphrase = passphrase
        self.demo_mode = demo_mode
        self._http_client = httpx.AsyncClient(base_url=OKX_REST_URL, timeout=timeout)
        self.ws_client: Optional[OKXPrivateWS] = None  # WebSocket客户端实例

        logger.info(f"OKX REST Client for account '{account_name}' initialized. Demo mode: {self.demo_mode}")

    @staticmethod
    def _get_timestamp() -> str:
        return datetime.utcnow().isoformat(timespec="milliseconds") + "Z"

    def _sign(self, timestamp: str, method: str, request_path: str, body: str = "") -> str:
        message = timestamp + method.upper() + request_path + body
        mac = hmac.new(self.secret, message.encode('utf-8'), hashlib.sha256)
        return base64.b64encode(mac.digest()).decode()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=0.5, max=5),
           retry_error_callback=lambda retry_state: logger.error(
               f"[{self.account_name}] API request failed after {retry_state.attempt_number} attempts: {retry_state.outcome.exception()}"))
    async def _request(
            self, method: str, path: str, params: Optional[Dict] = None, body: Optional[Dict] = None
    ) -> Dict[str, Any]:
        timestamp = self._get_timestamp()
        body_str = json.dumps(body) if body else ""

        query_string = urlencode(params or {})
        sign_path = path + ("?" + query_string if query_string else "")
        signature = self._sign(timestamp, method, sign_path, body_str)
        headers = {
            "OK-ACCESS-KEY": self.key,
            "OK-ACCESS-SIGN": signature,
            "OK-ACCESS-TIMESTAMP": timestamp,
            "OK-ACCESS-PASSPHRASE": self.passphrase,
            "Content-Type": "application/json",
        }
        if self.demo_mode:
            headers.update(_SIM_HEADER)

        url = path  # httpx的url参数不包含查询参数
        logger.debug(
            f"[{self.account_name}] Request: {method} {url} Params: {params} Body: {body_str[:200]}")  # 日志截断body

        response = await self._http_client.request(method, url, params=params, content=body_str if body else None,
                                                   headers=headers)

        logger.debug(f"[{self.account_name}] Response: {response.status_code} {response.text[:200]}")  # 日志截断响应

        if response.status_code >= 400 and response.status_code < 500:  # 客户端错误不重试
            try:
                data = response.json()
                err_msg = f"OKX API Client Error <{data.get('code', response.status_code)}>: {data.get('msg', response.text)}"
                logger.error(f"[{self.account_name}] {err_msg}")
                raise httpx.HTTPStatusError(err_msg, request=response.request, response=response)
            except json.JSONDecodeError:
                raise httpx.HTTPStatusError(f"OKX API Client Error {response.status_code}: {response.text}",
                                            request=response.request, response=response)

        response.raise_for_status()  # 对于 5xx 或其他未处理的 4xx 会抛出异常，由 tenacity 处理重试

        data = response.json()
        if data.get("code") != "0":
            err_msg = f"OKX API Error <{data['code']}>: {data['msg']}"
            logger.error(f"[{self.account_name}] {err_msg}")
            # 特定错误码处理，例如资金不足等不应重试
            if data['code'] in ["51004"]:  # Insufficient balance
                raise RuntimeError(err_msg)  # 阻止重试
            raise httpx.HTTPStatusError(err_msg, request=response.request, response=response)  # 其他错误让tenacity重试
        return data

    async def get_balance(self, currency: str) -> Dict[str, Any]:
        response_data = await self._request("GET", "/api/v5/account/balance", params={"ccy": currency.upper()})
        # 响应结构: {"code":"0","data":[{"adjEq":"","details":[{"availBal":"","availEq":"","cashBal":"","ccy":"BTC",...}]}]}
        details = response_data["data"][0]["details"]
        for bal_detail in details:
            if bal_detail["ccy"] == currency.upper():
                return {
                    "currency": currency.upper(),
                    "total": float(bal_detail.get("eq", bal_detail.get("cashBal", 0))),
                    # 'eq' for margin, 'cashBal' for funding/trading
                    "available": float(bal_detail.get("availBal", 0)),
                    "frozen": float(bal_detail.get("frozenBal", 0))
                }
        return {"currency": currency.upper(), "total": 0.0, "available": 0.0, "frozen": 0.0}

    async def get_all_balances(self) -> Dict[str, Dict[str, Any]]:
        response_data = await self._request("GET", "/api/v5/account/balance")
        all_balances = {}
        if response_data and response_data.get("data"):
            for account_type_balance in response_data["data"]:  # e.g. trading account, funding account
                for detail in account_type_balance.get("details", []):
                    ccy = detail["ccy"].upper()
                    cash_bal = float(detail.get("cashBal", 0))  # 适用于交易账户和资金账户
                    if cash_bal > 1e-8:  # 只记录有意义的余额
                        if ccy not in all_balances:
                            all_balances[ccy] = {"currency": ccy, "total": 0.0, "available": 0.0, "frozen": 0.0}
                        # 简单累加，实际中不同账户类型余额意义不同，需仔细处理
                        all_balances[ccy]["total"] += cash_bal
                        all_balances[ccy]["available"] += float(detail.get("availBal", 0))
                        all_balances[ccy]["frozen"] += float(detail.get("frozenBal", 0))
        return all_balances

    async def place_order(
            self, inst_id: str, side: str, ord_type: str, size: str,
            px: Optional[str] = None, td_mode: str = "cross", cl_ord_id: Optional[str] = None,
            **kwargs
    ) -> OrderResult:
        body = {
            "instId": inst_id, "tdMode": td_mode, "side": side.lower(),
            "ordType": ord_type.lower(), "sz": str(size)
        }
        if cl_ord_id: body["clOrdId"] = cl_ord_id
        if ord_type.lower() in ["limit", "post_only", "fok", "ioc"] and px:
            body["px"] = str(px)

        # 添加其他可能的kwargs到body，例如止盈止损
        # body.update({k: str(v) for k, v in kwargs.items() if v is not None})
        if kwargs.get("slOrdPx"): body["slOrdPx"] = str(kwargs["slOrdPx"])
        if kwargs.get("tpOrdPx"): body["tpOrdPx"] = str(kwargs["tpOrdPx"])
        # ...等等，根据OKX API文档添加

        try:
            response_data = await self._request("POST", "/api/v5/trade/order", body=body)
            order_data = response_data["data"][0]
            return OrderResult(
                ok=True,
                order_id=order_data.get("ordId"),
                client_order_id=order_data.get("clOrdId"),
                data=order_data,
                error_code=order_data.get("sCode"),  # 成功时 sCode 是 "0"
                error_message=order_data.get("sMsg")
            )
        except (httpx.HTTPStatusError, RuntimeError, RetryError) as e:  # RetryError from tenacity
            logger.error(f"[{self.account_name}] Place order failed for {inst_id}: {e}")
            error_code = None
            error_message = str(e)
            if hasattr(e, 'response') and e.response is not None:
                try:
                    err_data = e.response.json()
                    error_code = err_data.get("code")
                    error_message = err_data.get("msg", error_message)
                except json.JSONDecodeError:
                    pass  # 保持原始错误信息
            return OrderResult(ok=False, error_code=error_code, error_message=error_message,
                               data={"request_body": body})

    async def cancel_order(self, inst_id: str, ord_id: Optional[str] = None,
                           cl_ord_id: Optional[str] = None) -> OrderResult:
        if not ord_id and not cl_ord_id:
            return OrderResult(ok=False, error_message="Either ord_id or cl_ord_id must be provided.")
        body = {"instId": inst_id}
        if ord_id: body["ordId"] = ord_id
        if cl_ord_id: body["clOrdId"] = cl_ord_id
        try:
            response_data = await self._request("POST", "/api/v5/trade/cancel-order", body=body)
            order_data = response_data["data"][0]
            return OrderResult(ok=True, data=order_data, error_code=order_data.get("sCode"),
                               error_message=order_data.get("sMsg"))
        except Exception as e:
            logger.error(f"[{self.account_name}] Cancel order failed for {inst_id}, ordId={ord_id}: {e}")
            return OrderResult(ok=False, error_message=str(e))

    async def get_order_details(self, inst_id: str, ord_id: Optional[str] = None,
                                cl_ord_id: Optional[str] = None) -> OrderResult:
        if not ord_id and not cl_ord_id:
            return OrderResult(ok=False, error_message="Either ord_id or cl_ord_id must be provided.")
        params = {"instId": inst_id}
        if ord_id: params["ordId"] = ord_id
        if cl_ord_id: params["clOrdId"] = cl_ord_id
        try:
            response_data = await self._request("GET", "/api/v5/trade/order", params=params)
            # 成功时，data是一个列表，通常包含一个订单
            if response_data["data"]:
                order_data = response_data["data"][0]
                return OrderResult(ok=True, data=order_data, order_id=order_data.get("ordId"))
            else:  # 订单不存在
                return OrderResult(ok=False, error_code="ORDER_NOT_FOUND", error_message="Order not found")
        except Exception as e:
            logger.error(f"[{self.account_name}] Get order details failed for {inst_id}, ordId={ord_id}: {e}")
            return OrderResult(ok=False, error_message=str(e))

    async def get_positions(self, inst_id: Optional[str] = None, inst_type: str = "SWAP") -> List[Dict[str, Any]]:
        params = {"instType": inst_type}
        if inst_id:
            params["instId"] = inst_id
        try:
            response_data = await self._request("GET", "/api/v5/account/positions", params=params)
            # data 是一个持仓列表
            return response_data.get("data", [])
        except Exception as e:
            logger.error(f"[{self.account_name}] Get positions failed for {inst_id}: {e}")
            return []

    async def subscribe_private_data(self, ws_event_callback: Callable[[str, Dict[str, Any]], Awaitable[None]]):
        if not self.ws_client:
            self.ws_client = OKXPrivateWS(
                account_name=self.account_name,
                api_key=self.key,
                api_secret_str=self.secret.decode('utf-8'),  # WS需要str
                passphrase=self.passphrase,
                demo_mode=self.demo_mode,
                on_event_callback=ws_event_callback
            )
        # 启动WS在一个独立的任务中，允许它在后台运行并重连
        asyncio.create_task(self.ws_client.run_forever())
        logger.info(f"[{self.account_name}] Private WebSocket subscription task created.")

    async def close_connection(self) -> None:
        logger.info(f"[{self.account_name}] Closing OKX REST client http_client...")
        await self._http_client.aclose()
        if self.ws_client:
            logger.info(f"[{self.account_name}] Closing OKX WebSocket client...")
            await self.ws_client.close()
        logger.info(f"[{self.account_name}] OKX client connections closed.")


# --- 3. 私有 WebSocket (exchange/okx_ws.py) ---
WS_URL_DEMO = "wss://wspap.okx.com:8443/ws/v5/private?brokerId=9999"  # Demo盘地址
WS_URL_REAL = "wss://ws.okx.com:8443/ws/v5/private"  # 实盘地址 (注意官方文档可能会更新)


class OKXPrivateWS:
    def __init__(
            self,
            account_name: str,
            api_key: str,
            api_secret_str: str,  # 注意是str
            passphrase: str,
            demo_mode: bool,
            on_event_callback: Callable[[str, Dict[str, Any]], Awaitable[None]],  # account_name, event_data
    ):
        self.account_name = account_name
        self.key = api_key
        self.secret = api_secret_str.encode('utf-8')
        self.passphrase = passphrase
        self.url = WS_URL_DEMO if demo_mode else WS_URL_REAL
        self.demo_mode = demo_mode
        self.on_event_callback = on_event_callback
        self._ws_connection: Optional[websockets.WebSocketClientProtocol] = None
        self._stop_event = asyncio.Event()
        self._is_running = False

    def _get_signature(self, timestamp: str) -> str:
        message = timestamp + "GET" + "/users/self/verify"
        mac = hmac.new(self.secret, message.encode('utf-8'), hashlib.sha256)
        return base64.b64encode(mac.digest()).decode()

    async def _login(self, ws: websockets.WebSocketClientProtocol):
        timestamp = str(int(time.time()))
        login_payload = {
            "op": "login",
            "args": [{
                "apiKey": self.key,
                "passphrase": self.passphrase,
                "timestamp": timestamp,
                "sign": self._get_signature(timestamp),
            }]
        }
        if self.demo_mode:  # 对于模拟盘WS，OKX文档有时需要特定的登录参数或头部，但标准登录应该也工作
            # login_payload["args"][0]["simulatedTrading"] = "1" # 某些旧文档提及，当前文档似乎不需要
            pass

        await ws.send(json.dumps(login_payload))
        logger.info(f"[{self.account_name}] WS Login request sent.")

    async def _subscribe(self, ws: websockets.WebSocketClientProtocol):
        # 订阅订单、账户变动、持仓变动等
        # instType: SPOT, MARGIN, SWAP, FUTURES, OPTION
        # 对于SWAP和FUTURES，instFamily可以用来订阅一个系列，或者用instId订阅单个
        # 这里我们用 SPOT 和 SWAP 作为示例
        subs_args = [
            {"channel": "orders", "instType": "SWAP"},  # 订阅所有永续合约订单更新
            {"channel": "orders", "instType": "SPOT"},  # 订阅所有现货订单更新
            {"channel": "account", "ccy": ""},  # 订阅账户余额变动 (空ccy代表所有)
            {"channel": "positions", "instType": "SWAP", "instFamily": ""},  # 订阅所有永续合约持仓 (空instFamily代表所有)
            # {"channel": "positions", "instId": "BTC-USDT-SWAP"}, # 或者订阅特定交易对的持仓
        ]
        subscribe_payload = {"op": "subscribe", "args": subs_args}
        await ws.send(json.dumps(subscribe_payload))
        logger.info(f"[{self.account_name}] WS Subscription request sent for {len(subs_args)} channels.")

    async def _handle_message(self, message_str: str):
        if message_str == "pong":
            logger.debug(f"[{self.account_name}] WS Received pong.")
            return

        try:
            message_data = json.loads(message_str)
            event_type = message_data.get("event")
            arg = message_data.get("arg", {})
            channel = arg.get("channel")
            data_list = message_data.get("data", [])

            if event_type == "login":
                if message_data.get("code") == "0":
                    logger.info(f"[{self.account_name}] WS Login successful.")
                    await self._subscribe(self._ws_connection)  # type: ignore
                else:
                    logger.error(f"[{self.account_name}] WS Login failed: {message_data}")
                    # 登录失败可能需要关闭连接并重试
                    if self._ws_connection and not self._ws_connection.closed:
                        await self._ws_connection.close()
                    return  # 避免后续处理

            elif event_type == "subscribe":
                if message_data.get("code") == "0":
                    logger.info(f"[{self.account_name}] WS Subscribed to channel: {arg}")
                else:
                    logger.error(f"[{self.account_name}] WS Subscription failed for {arg}: {message_data}")

            elif event_type == "error":
                logger.error(f"[{self.account_name}] WS Error received: {message_data}")

            elif channel and data_list:  # 推送数据
                logger.debug(
                    f"[{self.account_name}] WS Data for channel '{channel}': {data_list[0] if data_list else 'empty'}")
                # 将 channel 和 data_list[0] (通常只有一个元素) 传递给回调
                # 回调函数负责处理不同channel的数据格式
                for item in data_list:  # 一个消息中可能有多条数据
                    await self.on_event_callback(self.account_name, {"channel": channel, **item})  # 将channel信息也传入
            else:
                logger.debug(f"[{self.account_name}] WS Received other message: {message_str[:200]}")

        except json.JSONDecodeError:
            logger.error(f"[{self.account_name}] WS Failed to decode JSON message: {message_str}")
        except Exception as e:
            logger.exception(f"[{self.account_name}] WS Error processing message: {message_str}")

    async def run_once(self):
        """尝试连接并处理消息，直到连接断开或发生错误。"""
        try:
            async with websockets.connect(self.url, ping_interval=25, ping_timeout=30) as ws:  # OKX建议ping间隔<=30s
                self._ws_connection = ws
                self._is_running = True
                logger.info(f"[{self.account_name}] WS Connected to {self.url}")
                await self._login(ws)  # 登录和订阅在 login 成功后进行

                while not self._stop_event.is_set():
                    try:
                        message = await asyncio.wait_for(ws.recv(), timeout=35.0)  # 略长于ping间隔
                        await self._handle_message(str(message))
                    except asyncio.TimeoutError:
                        logger.debug(f"[{self.account_name}] WS No message received, sending ping.")
                        await ws.ping()  # 主动发送 ping
                    except websockets.exceptions.ConnectionClosed as e:
                        logger.warning(f"[{self.account_name}] WS Connection closed: {e.code} {e.reason}")
                        break  # 外层循环会处理重连
                    except Exception as e:
                        logger.error(f"[{self.account_name}] WS Error during receive loop: {e}", exc_info=True)
                        break  # 发生错误，尝试重连
        except Exception as e:
            logger.error(f"[{self.account_name}] WS Connection attempt failed: {e}")
        finally:
            self._is_running = False
            self._ws_connection = None  # 清理连接对象

    async def run_forever(self):
        """持续运行，并在断开时自动重连。"""
        logger.info(f"[{self.account_name}] Starting WS run_forever loop.")
        self._stop_event.clear()
        reconnect_delay = 1
        while not self._stop_event.is_set():
            await self.run_once()
            if self._stop_event.is_set():
                break
            logger.info(f"[{self.account_name}] WS Reconnecting in {reconnect_delay}s...")
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, 30)  # 指数退避，最长30秒
        logger.info(f"[{self.account_name}] WS run_forever loop has stopped.")

    async def close(self):
        """停止并关闭WebSocket连接。"""
        logger.info(f"[{self.account_name}] WS Close requested.")
        self._stop_event.set()
        if self._ws_connection and not self._ws_connection.closed:
            logger.info(f"[{self.account_name}] WS Closing active connection.")
            await self._ws_connection.close()
        self._is_running = False
        logger.info(f"[{self.account_name}] WS Closed.")


# --- 4. 多账户路由器 (exchange/router.py) ---
class MultiAccountExchangeRouter:
    def __init__(self, account_credentials: Dict[str, Dict[str, str]], demo_mode: bool,
                 ws_event_callback: Optional[Callable[[str, Dict[str, Any]], Awaitable[None]]] = None):
        """
        account_credentials: {'account_name': {'api_key': ..., 'api_secret': ..., 'passphrase': ...}}
        demo_mode: True for simulated trading, False for live.
        ws_event_callback: Async callback for WebSocket private data events (account_name, event_data).
        """
        self.clients: Dict[str, OKXRestClient] = {}
        self.demo_mode = demo_mode
        self.ws_event_callback = ws_event_callback

        for name, creds in account_credentials.items():
            if not all([creds.get('api_key'), creds.get('api_secret'), creds.get('passphrase')]):
                logger.warning(f"Skipping account '{name}' due to missing credentials.")
                continue
            self.clients[name] = OKXRestClient(
                account_name=name,
                api_key=creds['api_key'],
                api_secret=creds['api_secret'],
                passphrase=creds['passphrase'],
                demo_mode=self.demo_mode
            )
            logger.info(f"Router initialized client for account '{name}'. Demo: {self.demo_mode}")

    async def initialize_all_ws(self):
        """为所有已配置的账户启动私有数据WebSocket订阅。"""
        if not self.ws_event_callback:
            logger.warning(
                "No WebSocket event callback provided to MultiAccountExchangeRouter. Skipping WS initialization.")
            return

        tasks = []
        for account_name, client_instance in self.clients.items():
            logger.info(f"Initiating private data subscription for account '{account_name}'...")
            # subscribe_private_data 现在会创建一个后台任务来运行 run_forever
            await client_instance.subscribe_private_data(self.ws_event_callback)
        # asyncio.gather(*tasks) # 不需要 gather，因为 subscribe_private_data 内部 create_task

    def get_client(self, account_name: str) -> Optional[OKXRestClient]:
        client = self.clients.get(account_name)
        if not client:
            logger.error(f"No client found for account name: '{account_name}'. Available: {list(self.clients.keys())}")
        return client

    async def close_all_connections(self):
        logger.info("Closing all exchange client connections...")
        tasks = [client.close_connection() for client in self.clients.values()]
        await asyncio.gather(*tasks, return_exceptions=True)  # 忽略单个关闭错误
        logger.info("All exchange client connections have been requested to close.")