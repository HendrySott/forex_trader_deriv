from .common import *

class DerivWebSocketAPI:
    """Deriv WebSocket API Client for Demo Account"""
    
    def __init__(self, app_id="1089", demo=True):
        self.app_id = app_id
        self.demo = demo
        self.websocket_url = f"wss://ws.binaryws.com/websockets/v3?app_id={app_id}"
        self.ws = None
        self.connected = False
        self.authenticated = False
        self.current_prices = {}
        self.balance = 10000
        self.account_info = {}
        self.running = False
        self.callbacks = {}
        self.last_buy = None
        self.last_error_message = None
        self.pending_requests = {}
        self.req_id_counter = 100
        self.contract_subscriptions = {}

    def _next_req_id(self):
        """Generate a unique request id for request/response flows."""
        self.req_id_counter += 1
        return self.req_id_counter

    def _store_response(self, data):
        """Store a response for any blocking request waiting on req_id."""
        req_id = data.get('req_id')
        if req_id is None:
            return
        pending = self.pending_requests.get(req_id)
        if pending:
            pending['response'] = data
            pending['event'].set()
        
    def on_message(self, ws, message):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(message)
            self._store_response(data)
            
            if 'error' in data:
                self.last_error_message = data['error']['message']
                log_event(f"API Error: {self.last_error_message}", logging.ERROR)
                
            elif 'msg_type' in data:
                msg_type = data['msg_type']
                
                if msg_type == 'tick':
                    tick_data = data.get('tick', {})
                    symbol = (
                        data.get('symbol')
                        or tick_data.get('symbol')
                        or data.get('echo_req', {}).get('ticks')
                    )
                    if not symbol or 'quote' not in tick_data or 'epoch' not in tick_data:
                        logger.warning("Skipping malformed tick payload: %s", data)
                        return
                    self.current_prices[symbol] = {
                        'price': tick_data['quote'],
                        'time': tick_data['epoch']
                    }
                    if 'tick' in self.callbacks:
                        for callback in self.callbacks['tick']:
                            callback(symbol, tick_data)
                
                elif msg_type == 'balance':
                    self.balance = data['balance']['balance']
                    if 'balance' in self.callbacks:
                        for callback in self.callbacks['balance']:
                            callback(data['balance'])
                
                elif msg_type == 'authorize':
                    if 'error' in data:
                        log_event(f"Authentication failed: {data['error']['message']}", logging.ERROR)
                        self.authenticated = False
                    else:
                        self.authenticated = True
                        self.account_info = data['authorize']
                        self.balance = data['authorize']['balance']
                        log_event("Connected to Deriv Demo Account")
                        log_event(f"   Account ID: {self.account_info.get('loginid')}")
                        log_event(f"   Balance: ${self.balance:.2f}")
                        if 'auth' in self.callbacks:
                            for callback in self.callbacks['auth']:
                                callback(self.account_info)
                
                elif msg_type == 'buy':
                    self.last_buy = data['buy']
                    log_event(f"Order executed. Contract ID: {data['buy']['contract_id']}")
                    if 'buy' in self.callbacks:
                        for callback in self.callbacks['buy']:
                            callback(data['buy'])

                elif msg_type == 'proposal_open_contract':
                    contract = data.get('proposal_open_contract', {})
                    contract_id = contract.get('contract_id')
                    if contract_id is not None:
                        self.contract_subscriptions[contract_id] = contract
                    if 'contract_update' in self.callbacks:
                        for callback in self.callbacks['contract_update']:
                            callback(contract)
                
                elif 'sell' in data:
                    log_event(f"Position closed. Contract: {data['sell']['contract_id']}")
                    if 'sell' in self.callbacks:
                        for callback in self.callbacks['sell']:
                            callback(data['sell'])
                        
        except Exception as e:
            logger.exception("Error processing message: %s", e)
    
    def on_error(self, ws, error):
        log_event(f"WebSocket error: {error}", logging.ERROR)
        self.connected = False
    
    def on_close(self, ws, close_status_code, close_msg):
        log_event("WebSocket connection closed")
        self.connected = False
        self.authenticated = False
    
    def on_open(self, ws):
        log_event("WebSocket connection opened")
        self.connected = True
    
    def connect(self):
        """Connect to Deriv WebSocket"""
        try:
            self.ws = websocket.WebSocketApp(
                self.websocket_url,
                on_open=self.on_open,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close
            )
            
            wst = threading.Thread(target=self.ws.run_forever, kwargs={
                'sslopt': {"cert_reqs": ssl.CERT_NONE}
            })
            wst.daemon = True
            wst.start()
            
            time.sleep(2)
            return self.connected
            
        except Exception as e:
            logger.exception("Failed to connect: %s", e)
            return False
    
    def authenticate(self, api_token):
        """Authenticate with API token"""
        if not self.connected:
            log_event("Not connected to WebSocket", logging.WARNING)
            return False
        
        auth_msg = {"authorize": api_token, "req_id": 1}
        self.ws.send(json.dumps(auth_msg))
        return True
    
    def subscribe_ticks(self, symbol, callback):
        """Subscribe to real-time ticks"""
        if not self.connected:
            return False
        
        if 'tick' not in self.callbacks:
            self.callbacks['tick'] = []
        self.callbacks['tick'].append(callback)
        
        subscribe_msg = {"ticks": symbol, "subscribe": 1, "req_id": 2}
        self.ws.send(json.dumps(subscribe_msg))
        return True

    def fetch_active_symbols(self, product_type="basic"):
        """Fetch the currently active symbols available from Deriv."""
        request = {
            "active_symbols": "brief",
            "req_id": self._next_req_id()
        }
        if product_type:
            request["product_type"] = product_type
        response = self.send_request_and_wait(request)
        return response.get("active_symbols", [])

    def subscribe_contract_updates(self, contract_id, callback):
        """Subscribe to updates for a specific open contract."""
        if not self.connected:
            return False

        if 'contract_update' not in self.callbacks:
            self.callbacks['contract_update'] = []
        if callback not in self.callbacks['contract_update']:
            self.callbacks['contract_update'].append(callback)

        subscribe_msg = {
            "proposal_open_contract": 1,
            "contract_id": contract_id,
            "subscribe": 1,
            "req_id": self._next_req_id()
        }
        self.ws.send(json.dumps(subscribe_msg))
        return True

    def sell_contract(self, contract_id, minimum_price=0):
        """Request an early sell/close for an open contract."""
        if not self.authenticated:
            log_event("Not authenticated", logging.WARNING)
            return None

        sell_msg = {
            "sell": int(contract_id),
            "price": float(minimum_price),
            "req_id": self._next_req_id()
        }
        try:
            sell_response = self.send_request_and_wait(sell_msg)
            sell_data = sell_response.get('sell')
            if not sell_data:
                raise RuntimeError("Deriv did not return sell confirmation data.")
            return sell_data
        except Exception as e:
            log_event(f"Sell request failed: {e}", logging.ERROR)
            return None

    def send_request_and_wait(self, payload, timeout=10):
        """Send a WebSocket request and wait for a response with the same req_id."""
        if not self.connected or not self.ws:
            raise RuntimeError("WebSocket is not connected.")

        req_id = payload.get("req_id", self._next_req_id())
        payload["req_id"] = req_id

        event = threading.Event()
        self.pending_requests[req_id] = {
            'event': event,
            'response': None
        }

        try:
            self.last_error_message = None
            self.ws.send(json.dumps(payload))
            if not event.wait(timeout):
                raise TimeoutError(f"Timed out waiting for Deriv response to req_id {req_id}.")

            response = self.pending_requests[req_id]['response']
            if response is None:
                raise RuntimeError("Received an empty response from Deriv.")
            if 'error' in response:
                self.last_error_message = response['error']['message']
                raise RuntimeError(self.last_error_message)
            return response
        finally:
            self.pending_requests.pop(req_id, None)
    
    def buy_contract(self, symbol, amount, contract_type, duration=1, duration_unit="h"):
        """Buy a contract by first requesting a proposal, then sending the buy request."""
        if not self.authenticated:
            log_event("Not authenticated", logging.WARNING)
            return False

        proposal_msg = {
            "proposal": 1,
            "amount": amount,
            "basis": "stake",
            "contract_type": contract_type,
            "currency": "USD",
            "duration": duration,
            "duration_unit": duration_unit,
            "symbol": symbol,
            "req_id": self._next_req_id()
        }

        try:
            proposal_response = self.send_request_and_wait(proposal_msg)
            proposal = proposal_response.get('proposal', {})
            proposal_id = proposal.get('id')
            ask_price = proposal.get('ask_price', amount)

            if not proposal_id:
                raise RuntimeError("Deriv did not return a proposal id.")

            buy_msg = {
                "buy": proposal_id,
                "price": ask_price,
                "req_id": self._next_req_id()
            }
            buy_response = self.send_request_and_wait(buy_msg)
            return buy_response.get('buy')
        except Exception as e:
            log_event(f"Buy request failed: {e}", logging.ERROR)
            return None

    def fetch_historical_candles(self, symbol=DEFAULT_SCALPING_SYMBOL, granularity=3600, count=3000, api_token=None):
        """Fetch historical candle data from Deriv over WebSocket."""
        ws = None
        try:
            ws = websocket.create_connection(
                self.websocket_url,
                sslopt={"cert_reqs": ssl.CERT_NONE},
                timeout=20
            )

            if api_token:
                ws.send(json.dumps({"authorize": api_token, "req_id": 1}))
                auth_response = json.loads(ws.recv())
                if 'error' in auth_response:
                    raise RuntimeError(auth_response['error']['message'])

            history_request = {
                "ticks_history": symbol,
                "adjust_start_time": 1,
                "count": count,
                "end": "latest",
                "granularity": granularity,
                "style": "candles",
                "req_id": 2
            }
            ws.send(json.dumps(history_request))
            response = json.loads(ws.recv())

            if 'error' in response:
                raise RuntimeError(response['error']['message'])

            candles = response.get('candles', [])
            if not candles:
                raise RuntimeError("No candle data returned from Deriv.")

            df = pd.DataFrame([
                {
                    'timestamp': datetime.fromtimestamp(candle['epoch']),
                    'open': float(candle['open']),
                    'high': float(candle['high']),
                    'low': float(candle['low']),
                    'close': float(candle['close']),
                    'volume': float(candle.get('volume', 100))
                }
                for candle in candles
            ])

            return df.sort_values('timestamp').reset_index(drop=True)
        finally:
            if ws is not None:
                ws.close()

    def fetch_account_status(self, api_token):
        """Fetch account details, including live balance, from Deriv."""
        ws = None
        try:
            ws = websocket.create_connection(
                self.websocket_url,
                sslopt={"cert_reqs": ssl.CERT_NONE},
                timeout=20
            )
            ws.send(json.dumps({"authorize": api_token, "req_id": 1}))
            response = json.loads(ws.recv())
            if 'error' in response:
                raise RuntimeError(response['error']['message'])

            account = response.get('authorize', {})
            if not account:
                raise RuntimeError("No account data returned from Deriv.")

            self.account_info = account
            self.balance = float(account.get('balance', self.balance))
            return account
        finally:
            if ws is not None:
                ws.close()

