import os
import time
import schedule
from openai import OpenAI
import ccxt
import pandas as pd
from datetime import datetime
import json
from dotenv import load_dotenv
import numpy as np

load_dotenv()

# 初始化DeepSeek客户端
deepseek_client = OpenAI(
    api_key=os.getenv('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com"
)

# 初始化OKX交易所
exchange = ccxt.okx({
    'options': {
        'defaultType': 'swap',  # OKX使用swap表示永续合约
    },
    'apiKey': os.getenv('OKX_API_KEY'),
    'secret': os.getenv('OKX_SECRET'),
    'password': os.getenv('OKX_PASSWORD'),  # OKX需要交易密码
})

# 交易参数配置
TRADE_CONFIG = {
    'symbol': 'BTC/USDT:USDT',  # OKX的合约符号格式
    'amount': 0.1,  # 交易张数 (每张=0.01 BTC)
    'leverage': 15,  # 杠杆倍数
    'timeframe': '5m',  # 使用5分钟K线
    'test_mode': False,  # 测试模式
}

# 全局变量存储历史数据
price_history = []
signal_history = []
position = None

# 添加token统计
token_stats = {
    'total_calls': 0,
    'total_tokens': 0,
    'total_cost': 0.0,
    'avg_tokens_per_call': 0
}


def update_token_stats(usage):
    """更新token统计"""
    global token_stats
    
    if hasattr(usage, 'total_tokens'):
        token_stats['total_calls'] += 1
        token_stats['total_tokens'] += usage.total_tokens
        token_stats['total_cost'] += usage.total_tokens * 0.000002  # 假设每token $0.0001
        token_stats['avg_tokens_per_call'] = token_stats['total_tokens'] / token_stats['total_calls']
        
        print(f"Token统计更新:")
        print(f"  总调用次数: {token_stats['total_calls']}")
        print(f"  总token数: {token_stats['total_tokens']}")
        print(f"  总成本: ¥{token_stats['total_cost']:.4f}")
        print(f"  平均每次: {token_stats['avg_tokens_per_call']:.0f} tokens")


def print_token_summary():
    """打印token使用摘要"""
    print("\n" + "="*50)
    print("Token使用摘要")
    print("="*50)
    print(f"总调用次数: {token_stats['total_calls']}")
    print(f"总token数: {token_stats['total_tokens']}")
    print(f"总成本: ${token_stats['total_cost']:.4f}")
    print(f"平均每次: {token_stats['avg_tokens_per_call']:.0f} tokens")
    print("="*50)


def calculate_smart_money_indicators(df):
    """计算聪明钱指标"""
    # 1. 成交量移动平均
    df['volume_ma_5'] = df['volume'].rolling(window=5).mean()
    df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
    
    # 2. 成交量比率 (当前成交量/平均成交量)
    df['volume_ratio'] = df['volume'] / df['volume_ma_20']
    
    # 3. 价格变化率
    df['price_change'] = df['close'].pct_change()
    
    # 4. 成交量加权平均价格 (VWAP)
    df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    
    # 5. 价格相对VWAP的位置
    df['price_vs_vwap'] = (df['close'] - df['vwap']) / df['vwap'] * 100
    
    # 6. 聪明钱流入指标 (价格上涨+高成交量)
    df['smart_money_flow'] = np.where(
        (df['close'] > df['close'].shift(1)) & (df['volume_ratio'] > 1.5),
        1,  # 聪明钱流入
        np.where(
            (df['close'] < df['close'].shift(1)) & (df['volume_ratio'] > 1.5),
            -1,  # 聪明钱流出
            0    # 无明确信号
        )
    )
    
    # 7. 支撑阻力位 (最近20根K线的最高最低价)
    df['resistance'] = df['high'].rolling(window=20).max()
    df['support'] = df['low'].rolling(window=20).min()
    
    return df


def setup_exchange():
    """设置交易所参数"""
    try:
        # OKX设置杠杆
        exchange.set_leverage(
            TRADE_CONFIG['leverage'],
            TRADE_CONFIG['symbol'],
            {'mgnMode': 'cross'}  # 全仓模式，也可用'isolated'逐仓
        )
        print(f"设置杠杆倍数: {TRADE_CONFIG['leverage']}x")

        # 获取余额
        try:
            balance = exchange.fetch_balance()
            
            # 安全地获取USDT余额
            if 'USDT' in balance and 'free' in balance['USDT']:
                usdt_balance = balance['USDT']['free']
                print(f"当前USDT余额: {usdt_balance:.2f}")
            else:
                print("无法获取USDT余额信息")
                print(f"可用币种: {list(balance.keys())}")
                
        except Exception as e:
            print(f"获取余额失败: {e}")
            return False

        return True
    except Exception as e:
        print(f"交易所设置失败: {e}")
        return False


def get_multi_timeframe_data():
    """获取多时间周期的K线数据"""
    try:
        # 获取不同时间周期的数据
        timeframes = ['5m', '15m', '1h']  # 5分钟、15分钟、1小时
        multi_data = {}
        
        for tf in timeframes:
            # 获取50根K线
            ohlcv = exchange.fetch_ohlcv(TRADE_CONFIG['symbol'], tf, limit=50)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # 计算聪明钱指标
            df = calculate_smart_money_indicators(df)
            
            current_data = df.iloc[-1]
            previous_data = df.iloc[-2] if len(df) > 1 else current_data
            
            multi_data[tf] = {
                'price': current_data['close'],
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'high': current_data['high'],
                'low': current_data['low'],
                'volume': current_data['volume'],
                'timeframe': tf,
                'price_change': ((current_data['close'] - previous_data['close']) / previous_data['close']) * 100,
                'kline_data': df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'volume_ratio', 'vwap', 'resistance', 'support']].tail(20).to_dict('records'),
                'all_data': df
            }
        
        return multi_data
    except Exception as e:
        print(f"获取多周期数据失败: {e}")
        return None


def get_btc_ohlcv():
    """获取BTC/USDT的K线数据（保持向后兼容）"""
    try:
        # 获取最近50根K线
        ohlcv = exchange.fetch_ohlcv(TRADE_CONFIG['symbol'], TRADE_CONFIG['timeframe'], limit=50)

        # 转换为DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # 计算聪明钱指标
        df = calculate_smart_money_indicators(df)

        current_data = df.iloc[-1]
        previous_data = df.iloc[-2] if len(df) > 1 else current_data

        return {
            'price': current_data['close'],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'high': current_data['high'],
            'low': current_data['low'],
            'volume': current_data['volume'],
            'timeframe': TRADE_CONFIG['timeframe'],
            'price_change': ((current_data['close'] - previous_data['close']) / previous_data['close']) * 100,
            'kline_data': df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'volume_ratio', 'vwap', 'resistance', 'support']].tail(20).to_dict('records'),
            'all_data': df
        }
    except Exception as e:
        print(f"获取K线数据失败: {e}")
        return None


def get_current_position():
    """获取当前持仓情况"""
    try:
        positions = exchange.fetch_positions([TRADE_CONFIG['symbol']])

        for pos in positions:
            if pos['symbol'] == TRADE_CONFIG['symbol']:
                contracts = float(pos['contracts']) if pos['contracts'] else 0

                if contracts > 0:
                    return {
                        'side': pos['side'],  # 'long' or 'short'
                        'size': contracts,
                        'entry_price': float(pos['entryPrice']) if pos['entryPrice'] else 0,
                        'unrealized_pnl': float(pos['unrealizedPnl']) if pos['unrealizedPnl'] else 0,
                        'leverage': float(pos['leverage']) if pos['leverage'] else TRADE_CONFIG['leverage'],
                        'symbol': pos['symbol']
                    }

        return None

    except Exception as e:
        print(f"获取持仓失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_current_orders():
    """获取当前挂单情况"""
    try:
        orders = exchange.fetch_open_orders(TRADE_CONFIG['symbol'])
        
        order_data = {
            'total_orders': len(orders),
            'buy_orders': [],
            'sell_orders': [],
            'order_summary': ''
        }
        
        for order in orders:
            order_info = {
                'id': order['id'],
                'side': order['side'],
                'type': order['type'],
                'amount': float(order['amount']),
                'price': float(order['price']) if order['price'] else None,
                'status': order['status'],
                'timestamp': order['timestamp']
            }
            
            if order['side'] == 'buy':
                order_data['buy_orders'].append(order_info)
            else:
                order_data['sell_orders'].append(order_info)
        
        # 构建挂单摘要
        if order_data['total_orders'] > 0:
            order_data['order_summary'] = f"当前有{order_data['total_orders']}个挂单: "
            if order_data['buy_orders']:
                order_data['order_summary'] += f"{len(order_data['buy_orders'])}个买单 "
            if order_data['sell_orders']:
                order_data['order_summary'] += f"{len(order_data['sell_orders'])}个卖单"
        else:
            order_data['order_summary'] = "当前无挂单"
        
        return order_data
        
    except Exception as e:
        print(f"获取挂单失败: {e}")
        return {
            'total_orders': 0,
            'buy_orders': [],
            'sell_orders': [],
            'order_summary': '获取挂单数据失败'
        }


def analyze_with_deepseek_multi_timeframe(multi_data):
    """使用聪明钱策略进行多周期分析"""
    
    # 构建多周期K线数据文本
    analysis_text = ""
    
    for tf, data in multi_data.items():
        kline_text = f"【{tf}周期最近20根K线数据】\n"
        for i, kline in enumerate(data['kline_data']):
            trend = "阳线" if kline['close'] > kline['open'] else "阴线"
            change = ((kline['close'] - kline['open']) / kline['open']) * 100
            
            # 成交量分析
            volume_status = ""
            if kline['volume_ratio'] > 2.0:
                volume_status = " (成交量激增)"
            elif kline['volume_ratio'] < 0.5:
                volume_status = " (成交量萎缩)"
            else:
                volume_status = " (成交量正常)"
            
            kline_text += f"K线{i + 1}: {trend} 开盘:{kline['open']:.2f} 收盘:{kline['close']:.2f} 涨跌:{change:+.2f}%{volume_status}\n"
            kline_text += f"  成交量:{kline['volume']:.2f} 最高:{kline['high']:.2f} 最低:{kline['low']:.2f}\n"
            kline_text += f"  VWAP:{kline['vwap']:.2f} 阻力位:{kline['resistance']:.2f} 支撑位:{kline['support']:.2f}\n"
        
        analysis_text += kline_text + "\n"
    
    # 构建聪明钱分析文本
    smart_money_analysis = "【聪明钱策略分析】\n"
    for tf, data in multi_data.items():
        df = data['all_data']
        current_price = data['price']
        
        # 获取最新数据
        latest = df.iloc[-1]
        
        smart_money_analysis += f"{tf}周期:\n"
        smart_money_analysis += f"  当前价格: ${current_price:.2f}\n"
        smart_money_analysis += f"  成交量比率: {latest['volume_ratio']:.2f}\n"
        smart_money_analysis += f"  VWAP: ${latest['vwap']:.2f}\n"
        smart_money_analysis += f"  价格相对VWAP: {latest['price_vs_vwap']:+.2f}%\n"
        smart_money_analysis += f"  关键阻力位: ${latest['resistance']:.2f}\n"
        smart_money_analysis += f"  关键支撑位: ${latest['support']:.2f}\n"
        smart_money_analysis += f"  聪明钱流向: {latest['smart_money_flow']}\n"
        
        # 成交量状态分析
        if latest['volume_ratio'] > 2.0:
            smart_money_analysis += f"  ⚠️ 成交量激增 - 大资金活动\n"
        elif latest['volume_ratio'] < 0.5:
            smart_money_analysis += f"  📉 成交量萎缩 - 观望情绪\n"
        else:
            smart_money_analysis += f"  📊 成交量正常\n"
        
        # 价格位置分析
        if current_price > latest['resistance']:
            smart_money_analysis += f"  🚀 价格突破阻力位\n"
        elif current_price < latest['support']:
            smart_money_analysis += f"  📉 价格跌破支撑位\n"
        else:
            smart_money_analysis += f"  📊 价格在支撑阻力区间内\n"
    
    # 添加上次交易信号
    signal_text = ""
    if signal_history:
        last_signal = signal_history[-1]
        signal_text = f"\n【上次交易信号】\n信号: {last_signal.get('signal', 'N/A')}\n信心: {last_signal.get('confidence', 'N/A')}"
        if 'entry_price' in last_signal:
            signal_text += f"\n入场价格: ${last_signal.get('entry_price', 'N/A')}"
        if 'stop_loss' in last_signal:
            signal_text += f"\n止损价格: ${last_signal.get('stop_loss', 'N/A')}"
        if 'take_profit' in last_signal:
            signal_text += f"\n止盈价格: ${last_signal.get('take_profit', 'N/A')}"

    # 添加当前持仓信息
    current_pos = get_current_position()
    position_text = "无持仓" if not current_pos else f"{current_pos['side']}仓, 数量: {current_pos['size']}, 盈亏: {current_pos['unrealized_pnl']:.2f}USDT"
    
    # 添加当前挂单信息
    current_orders = get_current_orders()
    orders_text = current_orders['order_summary']
    
    # 详细挂单信息
    detailed_orders = ""
    if current_orders['total_orders'] > 0:
        detailed_orders = "\n【当前挂单详情】\n"
        for order in current_orders['buy_orders']:
            detailed_orders += f"买单: {order['side']} {order['amount']} @ ${order['price']:.2f} ({order['type']})\n"
        for order in current_orders['sell_orders']:
            detailed_orders += f"卖单: {order['side']} {order['amount']} @ ${order['price']:.2f} ({order['type']})\n"

    # 构建提示词
    prompt = f"""
    你是一个专业的加密货币交易分析师，专注于聪明钱策略。请基于以下多周期BTC/USDT数据进行分析：

    {analysis_text}

    {smart_money_analysis}

    {signal_text}

    【当前持仓】
    - 当前持仓: {position_text}
    
    【当前挂单】
    - 挂单状态: {orders_text}
    {detailed_orders}

    【聪明钱策略分析要求】
    1. 基于5分钟、15分钟、1小时三个周期的聪明钱策略分析
    2. 重点关注5分钟周期的短期聪明钱活动
    3. 使用15分钟周期确认趋势方向
    4. 结合1小时周期判断大趋势背景
    5. 识别大资金流向和机构行为模式
    6. 分析关键支撑阻力位的有效性
    7. 结合成交量异常判断聪明钱动向
    8. 根据当前仓位，是否要减仓
    9. 考虑当前挂单情况，是否要重新挂单
    10. 基于技术分析给出具体的入场价格、止损价格、止盈价格

    【多周期分析重点】
    - 5分钟：捕捉短期聪明钱活动，快速反应
    - 15分钟：确认趋势方向，过滤噪音
    - 1小时：判断大趋势背景，避免逆势交易

    【价格建议要求】
    - 挂单价格：当信心不足时，给出具体的挂单价格（基于支撑阻力位等待更好价格）
    - 市价价格：当信心十足时，给出市价交易参考价格（立即成交）
    - 止损价格：基于关键支撑/阻力位设置，风险控制在3-5%
    - 止盈价格：基于风险回报比1:2以上设置
    - 所有价格必须是具体的数字，不要用"当前价格"等模糊表述

    请用以下JSON格式回复：
    {{
        "signal": "BUY|SELL|HOLD",
        "reason": "聪明钱分析理由",
        "limit_price": 挂单价格（信心不足时使用，等待更好价格）,
        "market_price": 市价参考价格（信心十足时使用，立即成交）,
        "stop_loss": 具体止损价格,
        "take_profit": 具体止盈价格,
        "confidence": "HIGH|MEDIUM|LOW",
        "smart_money_analysis": "聪明钱流向分析",
        "risk_reward_ratio": "风险回报比",
        "key_levels": "关键价位说明",
        "timeframe_analysis": "多周期分析说明",
        "order_suggestion": "挂单建议: PLACE_ORDER|HOLD|CANCEL_EXISTING",
        "order_reason": "挂单理由说明"
    }}
    """
    
    try:
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "您是一位专业的聪明钱策略分析师，专注于识别大资金流向和机构行为模式。请基于成交量、支撑阻力位和价格行为给出精准的交易建议，包括具体的入场价格、止损价格、止盈价格。所有价格必须是具体的数字。"},
                {"role": "user", "content": prompt}
            ],
            stream=False
        )

        # 添加token统计
        if hasattr(response, 'usage'):
            usage = response.usage
            print(f"本次Token消耗:")
            print(f"  输入: {usage.prompt_tokens} tokens")
            print(f"  输出: {usage.completion_tokens} tokens")
            print(f"  总计: {usage.total_tokens} tokens")
            print(f"  成本: ${usage.total_tokens * 0.000002:.4f}")
            
            # 更新全局统计
            update_token_stats(usage)

        # 安全解析JSON
        result = response.choices[0].message.content
        start_idx = result.find('{')
        end_idx = result.rfind('}') + 1
        if start_idx != -1 and end_idx != 0:
            json_str = result[start_idx:end_idx]
            signal_data = json.loads(json_str)
        else:
            print(f"无法解析JSON: {result}")
            return None

        # 保存信号到历史记录
        signal_data['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        signal_history.append(signal_data)
        if len(signal_history) > 30:
            signal_history.pop(0)

        return signal_data

    except Exception as e:
        print(f"DeepSeek分析失败: {e}")
        return None


def analyze_with_deepseek(price_data):
    """使用DeepSeek分析市场并生成交易信号（保持向后兼容）"""

    # 添加当前价格到历史记录
    price_history.append(price_data)
    if len(price_history) > 20:
        price_history.pop(0)

    # 构建K线数据文本
    kline_text = f"【最近20根{TRADE_CONFIG['timeframe']}K线数据】\n"
    for i, kline in enumerate(price_data['kline_data']):
        trend = "阳线" if kline['close'] > kline['open'] else "阴线"
        change = ((kline['close'] - kline['open']) / kline['open']) * 100
        
        # 成交量分析
        volume_status = ""
        if 'volume_ratio' in kline and kline['volume_ratio'] > 2.0:
            volume_status = " (成交量激增)"
        elif 'volume_ratio' in kline and kline['volume_ratio'] < 0.5:
            volume_status = " (成交量萎缩)"
        else:
            volume_status = " (成交量正常)"
        
        kline_text += f"K线{i + 1}: {trend} 开盘:{kline['open']:.2f} 收盘:{kline['close']:.2f} 涨跌:{change:+.2f}%{volume_status}\n"
        if 'volume_ratio' in kline:
            kline_text += f"  成交量比率:{kline['volume_ratio']:.2f} VWAP:{kline['vwap']:.2f}\n"
            kline_text += f"  阻力位:{kline['resistance']:.2f} 支撑位:{kline['support']:.2f}\n"

    # 构建技术指标文本
    if len(price_history) >= 5:
        closes = [data['price'] for data in price_history[-5:]]
        sma_5 = sum(closes) / len(closes)
        price_vs_sma = ((price_data['price'] - sma_5) / sma_5) * 100

        indicator_text = f"【技术指标】\n5周期均价: {sma_5:.2f}\n当前价格相对于均线: {price_vs_sma:+.2f}%"
        
        # 添加聪明钱分析
        if 'all_data' in price_data:
            latest_bb = price_data['all_data'].iloc[-1]
            indicator_text += f"\n成交量比率: {latest_bb['volume_ratio']:.2f}"
            indicator_text += f"\nVWAP: {latest_bb['vwap']:.2f}"
            indicator_text += f"\n价格相对VWAP: {latest_bb['price_vs_vwap']:+.2f}%"
    else:
        indicator_text = "【技术指标】\n数据不足计算技术指标"

    # 添加上次交易信号
    signal_text = ""
    if signal_history:
        last_signal = signal_history[-1]
        signal_text = f"\n【上次交易信号】\n信号: {last_signal.get('signal', 'N/A')}\n信心: {last_signal.get('confidence', 'N/A')}"

    # 添加当前持仓信息
    current_pos = get_current_position()
    position_text = "无持仓" if not current_pos else f"{current_pos['side']}仓, 数量: {current_pos['size']}, 盈亏: {current_pos['unrealized_pnl']:.2f}USDT"

    prompt = f"""
    你是一个专业的加密货币交易分析师。请基于以下BTC/USDT {TRADE_CONFIG['timeframe']}周期数据进行分析：

    {kline_text}

    {indicator_text}

    {signal_text}

    【当前行情】
    - 当前价格: ${price_data['price']:,.2f}
    - 时间: {price_data['timestamp']}
    - 本K线最高: ${price_data['high']:,.2f}
    - 本K线最低: ${price_data['low']:,.2f}
    - 本K线成交量: {price_data['volume']:.2f} BTC
    - 价格变化: {price_data['price_change']:+.2f}%
    - 当前持仓: {position_text}

    【分析要求】
    1. 基于{TRADE_CONFIG['timeframe']}K线趋势和聪明钱指标给出交易信号: BUY(买入) / SELL(卖出) / HOLD(观望)
    2. 简要分析理由（考虑趋势连续性、支撑阻力、成交量、聪明钱流向等因素）
    3. 基于技术分析建议合理的止损价位
    4. 基于技术分析建议合理的止盈价位
    5. 评估信号信心程度

    请用以下JSON格式回复：
    {{
        "signal": "BUY|SELL|HOLD",
        "reason": "分析理由",
        "stop_loss": 具体价格,
        "take_profit": 具体价格,
        "confidence": "HIGH|MEDIUM|LOW"
    }}
    """

    try:
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system",
                 "content": f"您是一位急需资金为母亲治疗癌症的交易员，你现在需要从加密货币交易市场赚取足够的金额治疗你的母亲，专注于{TRADE_CONFIG['timeframe']}周期聪明钱策略分析。请结合K线形态、成交量分析和聪明钱指标做出判断。"},
                {"role": "user", "content": prompt}
            ],
            stream=False
        )

        # 添加token统计
        if hasattr(response, 'usage'):
            usage = response.usage
            print(f"本次Token消耗:")
            print(f"  输入: {usage.prompt_tokens} tokens")
            print(f"  输出: {usage.completion_tokens} tokens")
            print(f"  总计: {usage.total_tokens} tokens")
            print(f"  成本: ¥{usage.total_tokens * 0.000002:.4f}")
            
            # 更新全局统计
            update_token_stats(usage)

        # 安全解析JSON
        result = response.choices[0].message.content
        start_idx = result.find('{')
        end_idx = result.rfind('}') + 1
        if start_idx != -1 and end_idx != 0:
            json_str = result[start_idx:end_idx]
            signal_data = json.loads(json_str)
        else:
            print(f"无法解析JSON: {result}")
            return None

        # 保存信号到历史记录
        signal_data['timestamp'] = price_data['timestamp']
        signal_history.append(signal_data)
        if len(signal_history) > 30:
            signal_history.pop(0)

        return signal_data

    except Exception as e:
        print(f"DeepSeek分析失败: {e}")
        return None


def cancel_old_stop_orders():
    """取消旧的止盈止损订单"""
    try:
        orders = exchange.fetch_open_orders(TRADE_CONFIG['symbol'])
        cancelled_count = 0
        for order in orders:
            if 'STOP' in order.get('tag', '') or 'TP' in order.get('tag', ''):
                exchange.cancel_order(order['id'], TRADE_CONFIG['symbol'])
                print(f"已取消旧止盈止损订单: {order['id']}")
                cancelled_count += 1
        if cancelled_count > 0:
            print(f"共取消了 {cancelled_count} 个旧止盈止损订单")
        return True
    except Exception as e:
        print(f"取消旧止盈止损订单失败: {e}")
        return False


def set_stop_loss_take_profit(signal_data, position_side):
    """设置止盈止损订单"""
    try:
        if 'stop_loss' not in signal_data or 'take_profit' not in signal_data:
            print("缺少止盈止损价格信息")
            return False
            
        stop_loss_price = signal_data['stop_loss']
        take_profit_price = signal_data['take_profit']
        
        # 先取消旧的止盈止损订单
        cancel_old_stop_orders()
        time.sleep(1)  # 等待取消完成
        
        if position_side == 'long':
            # 多头持仓：止损价格低于入场价，止盈价格高于入场价
            print(f"设置多头止盈止损: 止损${stop_loss_price:,.2f}, 止盈${take_profit_price:,.2f}")
            
            # 设置止损订单（卖出）
            stop_loss_order = exchange.create_limit_order(
                TRADE_CONFIG['symbol'],
                'sell',
                TRADE_CONFIG['amount'],
                stop_loss_price,
                params={'tag': 'f1ee03b510d5SUDE_STOP'}
            )
            print(f"止损订单设置成功: {stop_loss_order['id']}")
            
            # 设置止盈订单（卖出）
            take_profit_order = exchange.create_limit_order(
                TRADE_CONFIG['symbol'],
                'sell',
                TRADE_CONFIG['amount'],
                take_profit_price,
                params={'tag': 'f1ee03b510d5SUDE_TP'}
            )
            print(f"止盈订单设置成功: {take_profit_order['id']}")
            
        elif position_side == 'short':
            # 空头持仓：止损价格高于入场价，止盈价格低于入场价
            print(f"设置空头止盈止损: 止损${stop_loss_price:,.2f}, 止盈${take_profit_price:,.2f}")
            
            # 设置止损订单（买入）
            stop_loss_order = exchange.create_limit_order(
                TRADE_CONFIG['symbol'],
                'buy',
                TRADE_CONFIG['amount'],
                stop_loss_price,
                params={'tag': 'f1ee03b510d5SUDE_STOP'}
            )
            print(f"止损订单设置成功: {stop_loss_order['id']}")
            
            # 设置止盈订单（买入）
            take_profit_order = exchange.create_limit_order(
                TRADE_CONFIG['symbol'],
                'buy',
                TRADE_CONFIG['amount'],
                take_profit_price,
                params={'tag': 'f1ee03b510d5SUDE_TP'}
            )
            print(f"止盈订单设置成功: {take_profit_order['id']}")
            
        return True
        
    except Exception as e:
        print(f"设置止盈止损失败: {e}")
        return False


def execute_limit_order(signal_data):
    """执行挂单"""
    try:
        # 如果没有entry_price，尝试从limit_price获取
        if 'entry_price' not in signal_data or signal_data['entry_price'] is None:
            if 'limit_price' in signal_data and signal_data['limit_price'] is not None:
                signal_data['entry_price'] = signal_data['limit_price']
                print(f"使用挂单价格: ${signal_data['entry_price']:,.2f}")
            else:
                print("挂单价格无效，无法执行挂单")
                return False
            
        if signal_data['signal'] == 'BUY':
            print(f"挂买单: {TRADE_CONFIG['amount']} @ ${signal_data['entry_price']:,.2f}")
            order = exchange.create_limit_order(
                TRADE_CONFIG['symbol'],
                'buy',
                TRADE_CONFIG['amount'],
                signal_data['entry_price'],
                params={'tag': 'f1ee03b510d5SUDE'}
            )
            print(f"买单挂单成功: {order['id']}")
            
            # 设置止盈止损
            time.sleep(2)  # 等待订单确认
            set_stop_loss_take_profit(signal_data, 'long')
            
        elif signal_data['signal'] == 'SELL':
            print(f"挂卖单: {TRADE_CONFIG['amount']} @ ${signal_data['entry_price']:,.2f}")
            order = exchange.create_limit_order(
                TRADE_CONFIG['symbol'],
                'sell',
                TRADE_CONFIG['amount'],
                signal_data['entry_price'],
                params={'tag': 'f1ee03b510d5SUDE'}
            )
            print(f"卖单挂单成功: {order['id']}")
            
            # 设置止盈止损
            time.sleep(2)  # 等待订单确认
            set_stop_loss_take_profit(signal_data, 'short')
            
        return True
        
    except Exception as e:
        print(f"挂单失败: {e}")
        return False


def cancel_existing_orders():
    """取消现有挂单"""
    try:
        orders = exchange.fetch_open_orders(TRADE_CONFIG['symbol'])
        if orders:
            print(f"取消 {len(orders)} 个现有挂单...")
            for order in orders:
                exchange.cancel_order(order['id'], TRADE_CONFIG['symbol'])
                print(f"已取消挂单: {order['id']}")
            return True
        else:
            print("没有需要取消的挂单")
            return True
    except Exception as e:
        print(f"取消挂单失败: {e}")
        return False


def execute_trade(signal_data, price_data):
    """执行交易"""
    global position

    current_position = get_current_position()
    current_orders = get_current_orders()

    print(f"交易信号: {signal_data['signal']}")
    print(f"信心程度: {signal_data['confidence']}")
    print(f"理由: {signal_data['reason']}")
    
    # 显示价格信息
    if 'limit_price' in signal_data and signal_data['limit_price'] is not None:
        print(f"挂单价格: ${signal_data['limit_price']:,.2f}")
    if 'market_price' in signal_data and signal_data['market_price'] is not None:
        print(f"市价参考: ${signal_data['market_price']:,.2f}")
    if 'stop_loss' in signal_data and signal_data['stop_loss'] is not None:
        print(f"止损价格: ${signal_data['stop_loss']:,.2f}")
    if 'take_profit' in signal_data and signal_data['take_profit'] is not None:
        print(f"止盈价格: ${signal_data['take_profit']:,.2f}")
    if 'risk_reward_ratio' in signal_data and signal_data['risk_reward_ratio'] is not None:
        print(f"风险回报比: {signal_data['risk_reward_ratio']}")
    
    # 显示挂单建议
    if 'order_suggestion' in signal_data:
        print(f"挂单建议: {signal_data['order_suggestion']}")
        print(f"挂单理由: {signal_data.get('order_reason', 'N/A')}")
    
    print(f"当前持仓: {current_position}")
    print(f"当前挂单: {current_orders['order_summary']}")

    if TRADE_CONFIG['test_mode']:
        print("测试模式 - 仅模拟交易")
        return

    # 根据挂单建议执行操作
    if 'order_suggestion' in signal_data:
        if signal_data['order_suggestion'] == 'PLACE_ORDER':
            print("执行挂单...")
            execute_limit_order(signal_data)
        elif signal_data['order_suggestion'] == 'CANCEL_EXISTING':
            print("取消现有挂单...")
            cancel_existing_orders()
        elif signal_data['order_suggestion'] == 'HOLD':
            print("建议观望，不执行挂单")
        else:
            print("不执行挂单")
    else:
        # 根据信心程度选择交易方式
        if signal_data['confidence'] == 'HIGH' and 'market_price' in signal_data and signal_data['market_price'] is not None:
            print("信心十足，使用市价交易...")
            signal_data['entry_price'] = signal_data['market_price']
            execute_market_trade(signal_data, current_position)
        elif signal_data['confidence'] in ['MEDIUM', 'LOW'] and 'limit_price' in signal_data and signal_data['limit_price'] is not None:
            print("信心不足，使用挂单价格...")
            signal_data['entry_price'] = signal_data['limit_price']
            execute_limit_order(signal_data)
        else:
            print("使用传统市价交易逻辑...")
            execute_market_trade(signal_data, current_position)


def execute_market_trade(signal_data, current_position):
    """执行市价交易（原有逻辑）"""
    try:
        if signal_data['signal'] == 'BUY':
            if current_position and current_position['side'] == 'short':
                print("平空仓并开多仓...")
                # 平空仓
                exchange.create_market_order(
                    TRADE_CONFIG['symbol'],
                    'buy',
                    current_position['size'],
                    params={'reduceOnly': True, 'tag': 'f1ee03b510d5SUDE'}
                )
                time.sleep(1)
                # 开多仓
                exchange.create_market_order(
                    TRADE_CONFIG['symbol'],
                    'buy',
                    TRADE_CONFIG['amount'],
                    params={'tag': 'f1ee03b510d5SUDE'}
                )
            elif not current_position:
                print("开多仓...")
                exchange.create_market_order(
                    TRADE_CONFIG['symbol'],
                    'buy',
                    TRADE_CONFIG['amount'],
                    params={'tag': 'f1ee03b510d5SUDE'}
                )
            else:
                print("已持有多仓，无需操作")
                return

            print("订单执行成功")
            # 设置止盈止损
            time.sleep(2)  # 等待订单确认
            set_stop_loss_take_profit(signal_data, 'long')

        elif signal_data['signal'] == 'SELL':
            if current_position and current_position['side'] == 'long':
                print("平多仓并开空仓...")
                # 平多仓
                exchange.create_market_order(
                    TRADE_CONFIG['symbol'],
                    'sell',
                    current_position['size'],
                    params={'reduceOnly': True, 'tag': 'f1ee03b510d5SUDE'}
                )
                time.sleep(1)
                # 开空仓
                exchange.create_market_order(
                    TRADE_CONFIG['symbol'],
                    'sell',
                    TRADE_CONFIG['amount'],
                    params={'tag': 'f1ee03b510d5SUDE'}
                )
            elif not current_position:
                print("开空仓...")
                exchange.create_market_order(
                    TRADE_CONFIG['symbol'],
                    'sell',
                    TRADE_CONFIG['amount'],
                    params={'tag': 'f1ee03b510d5SUDE'}
                )
            else:
                print("已持有空仓，无需操作")
                return

            print("订单执行成功")
            # 设置止盈止损
            time.sleep(2)  # 等待订单确认
            set_stop_loss_take_profit(signal_data, 'short')

        elif signal_data['signal'] == 'HOLD':
            print("建议观望，不执行交易")
            return

        # 更新持仓信息
        time.sleep(2)
        position = get_current_position()
        print(f"更新后持仓: {position}")

    except Exception as e:
        print(f"订单执行失败: {e}")
        import traceback
        traceback.print_exc()


def trading_bot():
    """主交易机器人函数"""
    print("\n" + "=" * 60)
    print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 1. 获取多周期K线数据
    multi_data = get_multi_timeframe_data()
    if not multi_data:
        return

    # 显示各周期当前价格
    for tf, data in multi_data.items():
        print(f"{tf}周期BTC价格: ${data['price']:,.2f} (变化: {data['price_change']:+.2f}%)")

    # 2. 使用DeepSeek进行聪明钱策略分析
    signal_data = analyze_with_deepseek_multi_timeframe(multi_data)
    if not signal_data:
        return

    # 3. 执行交易
    execute_trade(signal_data, multi_data['5m'])  # 使用5分钟数据作为主要参考


def main():
    """主函数"""
    print("BTC/USDT OKX聪明钱策略自动交易机器人启动成功！")

    if TRADE_CONFIG['test_mode']:
        print("当前为模拟模式，不会真实下单")
    else:
        print("实盘交易模式，请谨慎操作！")

    print(f"主交易周期: {TRADE_CONFIG['timeframe']}")
    print("已启用聪明钱策略分析、多周期K线数据和持仓跟踪功能")
    print("分析周期: 5分钟、15分钟、1小时")
    print("K线数据: 每个周期50根K线，分析最近20根")
    print("策略重点: 成交量分析、支撑阻力位、聪明钱流向")

    # 设置交易所
    if not setup_exchange():
        print("交易所初始化失败，程序退出")
        return

    # 根据时间周期设置执行频率
    if TRADE_CONFIG['timeframe'] == '5m':
        schedule.every(5).minutes.do(trading_bot)
        print("执行频率: 每5分钟一次")
    elif TRADE_CONFIG['timeframe'] == '1h':
        schedule.every().hour.at(":01").do(trading_bot)
        print("执行频率: 每小时一次")
    elif TRADE_CONFIG['timeframe'] == '15m':
        schedule.every(15).minutes.do(trading_bot)
        print("执行频率: 每15分钟一次")
    else:
        schedule.every(5).minutes.do(trading_bot)
        print("执行频率: 每5分钟一次")

    # 立即执行一次
    trading_bot()

    # 循环执行
    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        print_token_summary()
        print("程序已停止")


if __name__ == "__main__":
    main()