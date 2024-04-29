import pandas as pd
import numpy as np
import datetime
import os
from tqdm import tqdm
from altaron.base.__base import AltaronBaseClass
from altaron.base.__strategy import TradingStrategy
from altaron.base.__data_processor import DataProcessor
from altaron.utils.vis import plot_growth_comparison

class BackTestEnvironment(AltaronBaseClass):

    def __init__(
            self,
            strategy: TradingStrategy,
            processor: DataProcessor,
            capital = 1000,
            min_bet_size = 0.01,
            max_bet_amount = 1000,
            fee = 0.0005,
            **kwargs
    ): 
        
        self.strategy = strategy
        self.processor = processor

        self.tickers = list(self.processor.data_dict.keys())
        self.max_ticker_lookback = max([
            self.processor.cfg[ticker]["feature_window"]
            for ticker in self.tickers
        ])

        self.capital = capital
        self.fee = fee

        assert(isinstance(min_bet_size, float) and min_bet_size >= 0. and min_bet_size <= 1.), "Wrong"
        assert(max_bet_amount > 0), "Wrong"

        self.min_bet_size = min_bet_size

        self.max_bet_amount = min(max_bet_amount, capital)
        if self.max_bet_amount >= self.capital:
            self.cash += self.max_bet_amount*self.fee

        self.min_bet_amount = self.max_bet_amount*self.min_bet_size

        self.reset()
    
    def reset(self):
        self.equity = self.capital
        self.cash = self.capital

        self.trades = {}
        self.actions = []
        
    def __can_enter(
            self,
            bet_amount
    ):

        if self.cash < bet_amount:
            return False
        
        if bet_amount < self.min_bet_amount:
            return False
        
        return True
    
    def __calc_bet_amount(
            self,
            bet_pct
    ):
        bet = self.max_bet_amount*bet_pct

        if bet >= self.cash:
            return self.cash
        
        return bet        

    def __log_action(
            self,
            ticker,
            side,
            qty,
            price,
            date,
            action="Entry"
    ):
        
        info = {
            "Ticker": ticker,
            "Side": side,
            "QTY": qty,
            "Price": price,
            "Fee": price*qty*self.fee,
            "Date": date,
            "Action": action
        }

        self.actions.append(info)

    def __enter_position(
            self,
            ticker,
            side,
            bet_size,
            price,
            date,
    ):

        info = {
            "side": 0,
            "size": 0.,
            "qty": 0.,
            "net_entry_price": None,
            "pnl_pct": 0.,
            "net_pnl": 0.,
            "time": 0,
            "current_price": price
        }

        bet_amount = self.__calc_bet_amount(bet_size)

        if not self.__can_enter(bet_amount):
            return info
        
        info["size"] = 1 if bet_amount == self.cash else bet_size
        
        qty = bet_amount/price
        fee = bet_amount*self.fee

        self.cash -= (bet_amount + fee)

        info["qty"] = qty
        info["side"] = side
        info["net_entry_price"] = price

        self.__log_action(ticker, side, qty, price, date, action="Entry")

        return info
    
    def __close_position(
            self,
            ticker,
            position,
            price,
            date,
            action="Exit"
    ):

        qty = position["qty"]
        side = position["side"]
        entry_price = position["net_entry_price"]

        info = {
            "side": 0,
            "size": 0.,
            "qty": 0.,
            "net_entry_price": None,
            "pnl_pct": 0.,
            "net_pnl": 0.,
            "time": 0,
            "current_price": price
        }
        
        cost = entry_price*qty
        value = price*qty

        pnl = side*(value - cost)

        exit_fee = value*self.fee

        self.cash += cost + pnl - exit_fee

        self.__log_action(ticker, -side, qty, price, date, action=action)

        return info

    def __reverse_position(
            self,
            ticker,
            position,
            new_bet_size,
            price,
            date
    ): 

        side = position["side"]
        qty = position["qty"]
        entry_price = position["net_entry_price"]

        info = self.__close_position(
            ticker,
            position,
            price,
            date
        )
        
        info = self.__enter_position(
            ticker,
            -side,
            new_bet_size,
            price,
            date
        )

        return info

    def __decrease_position_bet(
            self,
            ticker,
            position,
            price,
            new_size,
            date,
    ):  
        
        side = position["side"]
        qty = position["qty"]
        current_size = position["size"]
        entry_price = position["net_entry_price"]
        trade_time = position["time"]
    
        if new_size < self.min_bet_size:
            return self.__close_position(
                ticker,
                position,
                price,
                date
            )

        new_qty = (new_size/current_size)*qty
        close_qty = qty - new_qty

        sell_position = position.copy()
        sell_position["qty"] = close_qty

        _ = self.__close_position(
            ticker,
            sell_position,
            price,
            date
        )

        info = {
            "side": side,
            "size": new_size,
            "qty": new_qty,
            "net_entry_price": entry_price,
            "pnl_pct": 0.,
            "net_pnl": 0.,
            "time": trade_time,
            "current_price": price
        }

        return info
    
    def __increase_position_bet(
            self,
            ticker,
            position,
            increase_size,
            price,
            date
    ):

        side = position["side"]
        qty = position["qty"]
        current_size = position["size"]
        entry_price = position["net_entry_price"]
        trade_time = position["time"]

        info = {
            "side": side,
            "size": current_size,
            "qty": qty,
            "side": side,
            "net_entry_price": entry_price,
            "pnl_pct": 0.,
            "net_pnl": 0.,
            "time": trade_time,
            "current_price": price
        }
        
        if current_size + increase_size >= 1:
            increase_size = 1 - current_size

        if increase_size < self.min_bet_amount:
            return info

        new_info = self.__enter_position(
            ticker,
            side,
            increase_size,
            price,
            date             
        )

        info["size"] += increase_size
        info["qty"] += new_info["qty"]

        net_qty = info["qty"]
        net_entry = (qty*entry_price + new_info["qty"]*price)/net_qty

        info["net_entry_price"] = net_entry

        return info
    
    def __update_position(
            self,
            position:dict,
            current_price
    ):

        info = position.copy()
        info["current_price"] = current_price

        if info["side"] == 0:
            return info
        
        info["time"] += 1

        s = info["side"]
        q = info["qty"]
        ent = info["net_entry_price"]
        
        info["net_pnl"] = s*q*(current_price - ent) - q*ent*self.fee
        info["pnl_pct"] = s*100*(current_price/ent - 1)
        
        return info
    
    def __update_equity(
            self,
            ticker_positions: dict,
    ):

        self.equity = 0

        for ticker, vals in ticker_positions.items():
            net_entry = vals["net_entry_price"]

            if net_entry is None:
                continue

            qty = vals["qty"]
            pnl = vals["pnl_pct"]

            self.equity += net_entry*qty*(pnl/100 + 1)
        
        self.equity += self.cash
    
    def __handle_stop_exit(
            self,
            ticker,
            date,
            position: dict,
            limits: dict,
            ohlcv
    ):
        side = position["side"]

        if side == 0:
            return position.copy(), False
        
        ll = limits["lower_limit"]
        ul = limits["upper_limit"]
        tl = limits["time_limit"]

        if ll is not None and ohlcv["Low"] <= ll:
            
            if side*(ll-position["net_entry_price"]) > 0:
                action = "Take Profit"
            else:
                action = "Stop Loss"

            return self.__close_position(
                ticker=ticker,
                position=position,
                price=ll,
                date=date,
                action=action
            ), True

        if ul is not None and ohlcv["High"] >= ul:
            if side*(ul-position["net_entry_price"]) > 0:
                action = "Take Profit"
            else:
                action = "Stop Loss"

            return self.__close_position(
                ticker=ticker,
                position=position,
                price=ul,
                date=date,
                action=action
            ), True
        
        if tl is not None and position["time"] >= tl:

            return self.__close_position(
                ticker=ticker,
                position=position,
                price=ohlcv["Close"],
                date=date,
                action="Time-Out"
            ), True

        return position.copy(), False

    def __handle_strategy_decision__(
            self,
            ticker,
            date,
            positions,
            strategy_outs
    ):
         
        pos = positions[ticker].copy()
        out = strategy_outs[ticker].copy()

        decisions = out["decision"]
        limits = out["limits"]

        ohlcv = self.processor.get_ticker_ohlcv(
            ticker,
            date,
        )

        pos, exit = self.__handle_stop_exit(
            ticker=ticker,
            date=date,
            position=pos,
            limits=limits,
            ohlcv=ohlcv
        )

        if exit:
            return pos

        side = pos["side"]
        size = pos["size"]

        decision_side = decisions["side"]
        decision_size = decisions["size"]

        if side == 0:
            if decision_side == 0:
                return pos
            elif decision_size < self.min_bet_size:
                return pos
            else:
                pos = self.__enter_position(
                    ticker=ticker,
                    side=decision_side,
                    bet_size=decision_size,
                    price=ohlcv["Close"],
                    date=date
                )
        
        #Decrease bet or possibly close trade
        #Since positions are fed to the strategy class;
        #Decision size is expected to represent size to exit
        #If decision size is greater than size - self.min_bet_size;
        #Position is closed
        #Else, position's bet is decreased by decision size
        elif decision_side == 0:
            pos = self.__decrease_position_bet(
                ticker=ticker,
                position=pos,
                price=ohlcv["Close"],
                new_size=size-decision_size,
                date=date
            )
        
        #Since positions are fed to the strategy class;
        #Decision size is expected to represent size to increase
        #If decision size+size is greater than 1
        #Position is increased by 1 - size
        #Else, position's bet is increased by decision size
        #If the increase size is lower than self.min_bet_size
        #Nothing happens
        elif decision_side == side:
            pos = self.__increase_position_bet(
                ticker=ticker,
                position=pos,
                increase_size=decision_size,
                price=ohlcv["Close"],
                date=date
            )

        #Reverse the position
        #Decision size represents the size of the new position
        #If decision size is smaller than self.min_bet_amount
        #Closes the current position
        elif decision_side != side:
            if decision_size < self.min_bet_size:
                pos = self.__close_position(
                    ticker=ticker,
                    position=pos,
                    price=ohlcv["Close"],
                    date=date
                )
            else:
                pos = self.__reverse_position(
                    ticker=ticker,
                    position=pos,
                    new_bet_size=decision_size,
                    price=ohlcv["Close"],
                    date=date
                )

        return pos
    
    def run_backtest(
            self,
            date_inputs,
            start_index,
            end_index
    ):

        self.equity_series = [self.equity for x in range(start_index)]
        
        ticker_positions = {
            ticker: {
                "side": 0,
                "size": 0.,
                "qty": 0.,
                "net_entry_price": None,
                "pnl_pct": 0.,
                "net_pnl": 0.,
                "time": 0,
                "current_price": None
            }
            for ticker in self.tickers
        }

        print("Starting Backtest...")
        for date in tqdm(list(date_inputs.keys())):
            
            ohlcv = self.processor.get_ohlcv(date)

            for ticker in self.tickers:
                ticker_positions[ticker] = self.__update_position(
                    position=ticker_positions[ticker],
                    current_price=ohlcv[ticker]["Close"]
                )

            self.__update_equity(ticker_positions)
            self.equity_series.append(self.equity)
   
            strategy_outs = self.strategy.get_strategy_out(
                date_inputs[date], ticker_positions, ohlcv
            )

            for ticker in self.tickers:

                pos = self.__handle_strategy_decision__(
                    ticker=ticker,
                    date=date,
                    positions=ticker_positions, 
                    strategy_outs=strategy_outs
                )

                ticker_positions[ticker].update(pos)

        self.equity_series = pd.Series(
                self.equity_series, 
                index=self.processor.data_dict[self.tickers[0]].index[:end_index]
        )

        return self.__report_backtest()
    
    def __report_backtest(
            self
    ):

        ticker_groupted_actions = {
            ticker: [
                action for action in self.actions
                if action["Ticker"] == ticker
            ]
            for ticker in self.tickers
        }

        ticker_tracking = {
            ticker: {
                "in_trade": False,
                "last_trade": 0,
                "start_side": 0,
                "net_entry_price": 0,
                "net_trade_qty": 0,
                "start_date": 0,
                "current_qty": 0,
                "net_exit_price": 0,
            }
            for ticker in self.tickers
        }

        if self.actions == []:
            return pd.DataFrame(), pd.DataFrame()

        for action in self.actions:

            ticker = action["Ticker"]
            in_trade = ticker_tracking[ticker]["in_trade"]

            action_side = action["Side"]
            action_qty = action["QTY"]
            action_price = action["Price"]

            if in_trade:
                
                trade_side = ticker_tracking[ticker]["start_side"]
                net_entry = ticker_tracking[ticker]["net_entry_price"]
                net_qty = ticker_tracking[ticker]["net_trade_qty"]

                #Increment trade size
                if trade_side == action_side:
                    
                    ticker_tracking[ticker]["net_entry_price"] = (net_entry*net_qty + action_price*action_qty)/(net_qty+action_qty)

                    ticker_tracking[ticker]["current_qty"] += action_qty
                    ticker_tracking[ticker]["net_trade_qty"] += action_qty
                
                #Decrement, exit or reverse trade
                elif trade_side != action_side:
                    
                    cur_qty = ticker_tracking[ticker]["current_qty"]

                    net_exit = ticker_tracking[ticker]["net_exit_price"]                        
                    net_exited_qty = ticker_tracking[ticker]["net_trade_qty"] - cur_qty

                    #Decrement
                    if cur_qty > action_qty:
                    
                        ticker_tracking[ticker]["net_exit_price"] = (net_exit*net_exited_qty + action_price*action_qty)/(net_exited_qty+action_qty)
                        ticker_tracking[ticker]["current_qty"] -= action_qty
                    
                    #Close Trade
                    elif cur_qty == action_qty:
                        trade_exit_price = (net_exit*net_exited_qty + action_price*action_qty)/(net_exited_qty+action_qty)

                        pnl = trade_side*net_qty*(trade_exit_price-net_entry)
                        pnl_pct = trade_side*100*(trade_exit_price/net_entry - 1)
                        total_fee = net_qty*self.fee*(net_entry+trade_exit_price)

                        trade_info = {
                            "Ticker": ticker,
                            "Trade Side": trade_side,
                            "Net QTY": net_qty,
                            "Net Entry": net_entry,
                            "Net Exit": trade_exit_price,
                            "PnL": pnl,
                            "PnL%": pnl_pct,
                            "Fee": total_fee,
                            "Entry Date": ticker_tracking[ticker]["start_date"],
                            "Exit Date": action["Date"],
                            "Exit Type": action["Action"]                 
                        }

                        lt = ticker_tracking[ticker]["last_trade"]
                        self.trades[f"Trade{lt}"] = trade_info
                        
                        ticker_tracking[ticker] = {
                            "in_trade": False,
                            "last_trade": lt,
                            "start_side": 0,
                            "net_entry_price": 0,
                            "net_trade_qty": 0,
                            "start_date": 0,
                            "current_qty": 0,
                            "net_exit_price": 0,
                        }

                    #Reverse
                    elif cur_qty < action_qty:
                        #action_qty = cur_qty + new_position_qty

                        trade_exit_price = (net_exit*net_exited_qty + action_price*cur_qty)/(net_exited_qty+cur_qty)

                        pnl = trade_side*net_qty*(trade_exit_price-net_entry)
                        pnl_pct = trade_side*100*(trade_exit_price/net_entry - 1)
                        total_fee = net_qty*self.fee*(net_entry+trade_exit_price)

                        trade_info = {
                            "Ticker": ticker,
                            "Trade Side": trade_side,
                            "Net QTY": net_qty,
                            "Net Entry": net_entry,
                            "Net Exit": trade_exit_price,
                            "PnL": pnl,
                            "PnL%": pnl_pct,
                            "Fee": total_fee,
                            "Entry Date": ticker_tracking[ticker]["start_date"],
                            "Exit Date": action["Date"]                           
                        }

                        lt = ticker_tracking[ticker]["last_trade"]
                        self.trades[f"Trade{lt}"] = trade_info

                        new_position_qty = action_qty - cur_qty

                        ticker_tracking[ticker] = {
                            "in_trade": True,
                            "last_trade": lt+1,
                            "start_side": action_side,
                            "net_entry_price": action_price,
                            "net_trade_qty": new_position_qty,
                            "start_date": action["Date"],
                            "current_qty": new_position_qty,
                            "net_exit_price": 0,
                        }

            elif not in_trade:
                lt = ticker_tracking[ticker]["last_trade"]

                ticker_tracking[ticker] = {
                    "in_trade": True,
                    "last_trade": lt+1,
                    "start_side": action_side,
                    "net_entry_price": action_price,
                    "net_trade_qty": action_qty,
                    "start_date": action["Date"],
                    "current_qty": action_qty,
                    "net_exit_price": 0,
                }

        trade_df = pd.DataFrame(self.trades).transpose()

        actions = pd.DataFrame(
            data=np.array([list(a.values()) for a in self.actions]), 
            index=[f"Action{x}" for x in range(len(self.actions))],
            columns=list(self.actions[0].keys())
        )

        evaluator = BackTestEvaluator(trade_df, actions, self.equity_series)

        #This is wrong implementation
        #tickers = {
        #    ticker: self.processor.data_dict[ticker]["Close"].loc[self.equity_series.index[0]:self.equity_series.index[-1]]
        #    for ticker in self.tickers
        #}

        #evaluator.plot_growth(tickers)        
        
        stats = evaluator.report_stats()

        self.reset()

        return trade_df, actions, stats

class BackTestEvaluator(AltaronBaseClass):

    def __init__(
            self,
            trades: pd.DataFrame,
            actions: pd.DataFrame,
            equity_arr: pd.Series,
    ):
        
        self.trades = trades
        self.actions = actions

        self.equity_arr = equity_arr
        self.capital = equity_arr.iloc[0]

        self.total_days = (self.actions["Date"].max() - self.actions["Date"].min()).days
        self.periods = "Daily"
        if self.total_days == 0:
            #Get minutes instead
            self.total_days = (self.actions["Date"].max() - self.actions["Date"].min()).seconds/60
            self.periods = "Minutely"
    
    def plot_growth(
            self,
            tickers,
            figsize=(16,6),
            save_to=None
    ):
        
        equities = tickers.copy()
        equities["Equity"] = self.equity_arr
        print(equities)
        plot_growth_comparison(
            equities=equities,
            figsize=figsize,
            normalized=True,
            save_to=save_to
        )
        
    
    def report_stats(self):

        if len(self.trades["Ticker"].unique()) == 1:
            ticker = self.trades["Ticker"].iloc[0]
            index = []
            all_stats = {ticker: {}}

            cs = self.characteristic_stats()
            all_stats[ticker].update(cs)
            index.extend(list(cs.keys()))

            ps = self.performance_stats()
            all_stats[ticker].update(ps)
            index.extend(list(ps.keys()))

            rs = self.run_stats()
            all_stats[ticker].update(rs)
            index.extend(list(rs.keys()))

            return pd.DataFrame(all_stats).loc[index]
        
        else:
            og_trades = self.trades.copy()
            og_actions = self.actions.copy()

            index = []
            all_stats = {"all": {}}

            cs = self.characteristic_stats()
            all_stats["all"].update(cs)
            index.extend(list(cs.keys()))

            ps = self.performance_stats()
            all_stats["all"].update(ps)
            index.extend(list(ps.keys()))

            rs = self.run_stats()
            all_stats["all"].update(rs)
            index.extend(list(rs.keys()))

            for ticker in self.trades["Ticker"].unique():
                
                all_stats[ticker] = {}
                
                self.trades = og_trades[og_trades["Ticker"] == ticker].copy()
                self.actions = og_actions[og_actions["Ticker"] == ticker].copy()

                cs = self.characteristic_stats()
                all_stats[ticker].update(cs)

                ps = self.performance_stats()
                all_stats[ticker].update(ps)

                rs = self.run_stats()
                all_stats[ticker].update(rs)
            
            self.trades = og_trades.copy()
            self.actions = og_actions.copy()
        
            return pd.DataFrame(all_stats).loc[index]

    def characteristic_stats(self):

        stats = {}

        stats["Test Time"] = str(self.actions["Date"].max() - self.actions["Date"].min())
        stats["Avg Holding Time"] = str((self.trades["Exit Date"] - self.trades["Entry Date"]).mean())
        stats[f"Avg {self.periods} Trades"] = round(len(self.trades)/self.total_days, 2)
        stats["Long Ratio"] = round(sum(self.trades["Trade Side"] == 1)/len(self.trades), 2)

        return stats

    def performance_stats(self):
        
        stats = {}

        stats["Total PnL"] = self.trades["PnL"].sum()
        stats["Total PnL%"] = self.trades["PnL%"].sum()
        stats["Total Fee Paid"] = self.trades["Fee"].sum()

        stats["Net Gain"] = stats["Total PnL"] - stats["Total Fee Paid"]
        stats["Net Gain%"] = 100*((stats["Net Gain"] + self.capital)/self.capital - 1)

        stats["Long Position Total PnL"] = self.trades[self.trades["Trade Side"] == 1]["PnL"].sum()
        
        stats[f"Avg {self.periods} Return"] = round(stats["Net Gain"]/self.total_days, 2)
        stats[f"Avg {self.periods} Return%"] = round(stats["Net Gain%"]/self.total_days, 2)

        stats["Win Rate"] = len(self.trades[self.trades["PnL"] > 0])/len(self.trades)
        stats["Long Win Rate"] = len(self.trades[
            (self.trades["PnL"] > 0) & 
            (self.trades["Trade Side"] == 1)
        ])/sum(self.trades["Trade Side"] == 1) 
        
        stats["Avg PnL"] = round(self.trades["PnL"].mean(), 2)
        stats["Avg PnL%"] = round(self.trades["PnL%"].mean(), 2)

        try:
            stats["Avg Profit"] = self.trades[self.trades["PnL"] > 0]["PnL"].sum()/len(self.trades[self.trades["PnL"] > 0])
            stats["Avg Profit%"] = self.trades[self.trades["PnL%"] > 0]["PnL%"].sum()/len(self.trades[self.trades["PnL%"] > 0])
        except:
            stats["Avg Profit"] = 0
            stats["Avg Profit%"] = 0

        try:
            stats["Avg Loss"] =  self.trades[self.trades["PnL"] < 0]["PnL"].sum()/len(self.trades[self.trades["PnL"] < 0])
            stats["Avg Loss%"] =  self.trades[self.trades["PnL%"] < 0]["PnL%"].sum()/len(self.trades[self.trades["PnL%"] < 0])
        except:
            stats["Avg Loss"] = 0
            stats["Avg Loss%"] = 0
        
        return stats

    def run_stats(self):

        stats = {}

        return_arr = self.trades["PnL"].values

        stats["HHI Positive"] = self.__get_herfindahl_hirschman_index(return_arr[return_arr >= 0])
        stats["HHI Negative"] = self.__get_herfindahl_hirschman_index(return_arr[return_arr < 0])

        dd_tuw = self.__get_dd_and_tuw()

        stats["Max Drawdown"] = dd_tuw["DD"].max()
        stats["Max Drawdown%"] = 100*(dd_tuw["HWM"].iloc[dd_tuw["DD"].argmax()] - stats["Max Drawdown"])/dd_tuw["HWM"].iloc[dd_tuw["DD"].argmax()]

        stats["Highest Gain"] = dd_tuw["HWM"].max() - self.capital
        stats["Highest Gain%"] = 100*(stats["Highest Gain"]/self.capital)

        return stats
    
    def __get_dd_and_tuw(self):

        df = self.equity_arr.copy().to_frame("Equity")
        df["HWM"] = df["Equity"].expanding().max()
        df["DD"] = df["HWM"] - df["Equity"]

        t = 0
        tuw = []

        for i in range(len(df)):

            if df["DD"].iloc[i] > 0:
                t += 1
            else:
                t = 0
            
            tuw.append(t)

        df["TUW"] = tuw

        return df

    def __get_herfindahl_hirschman_index(self, returns: np.ndarray):

        if len(returns) < 2:
            return np.nan
        
        size = len(returns)
        
        w = returns/returns.sum()
        ss_w = np.sum(np.square(w))

        concentration = (ss_w - 1/size)/(1 - 1/size)

        return concentration

