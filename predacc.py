import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

class ETHKuCoinPredictionAccuracyTracker:
    def __init__(self, patterns_file='eth_kucoin_patterns_parallel_3_4_5.csv'):
        self.base_url = "https://api-futures.kucoin.com"  # KuCoin Futures API
        self.symbol = "ETHUSDTM"  # ETH perpetual futures
        self.patterns_df = None
        
        # Prediction tracking
        self.prediction_history = []
        self.total_predictions = 0
        self.correct_predictions = 0
        self.wrong_predictions = 0
        self.current_prediction = None
        
        # Trading parameters
        self.min_confidence = 65
        self.min_occurrences = 15
        
        self.load_kucoin_patterns(patterns_file)
    
    def load_kucoin_patterns(self, patterns_file):
        """Load KuCoin-trained patterns"""
        try:
            self.patterns_df = pd.read_csv(patterns_file)
            print(f"✅ Loaded {len(self.patterns_df)} KuCoin ETH patterns (sizes 3,4,5)")
            
            # Show breakdown by combination size
            for size in [3, 4, 5]:
                size_patterns = self.patterns_df[self.patterns_df['combination_size'] == size]
                print(f"   Size {size}: {len(size_patterns)} patterns")
            
            # Filter for high confidence patterns
            high_confidence = self.patterns_df[
                (self.patterns_df['combination_size'].isin([3, 4, 5])) &
                (self.patterns_df['occurrences'] >= self.min_occurrences) &
                ((self.patterns_df['rise_percentage'] >= self.min_confidence) |
                 (self.patterns_df['fall_percentage'] >= self.min_confidence))
            ]
            
            print(f"📊 High confidence patterns: {len(high_confidence)}")
            print(f"🎯 Confidence Threshold: {self.min_confidence}%+")
            print(f"🏢 Using KuCoin Futures API")
            return True
            
        except FileNotFoundError:
            print(f"❌ Pattern file {patterns_file} not found!")
            print("💡 Make sure to run the KuCoin training system first to generate patterns")
            return False
    
    def get_available_eth_symbols(self):
        """Get available ETH futures symbols from KuCoin"""
        try:
            url = f"{self.base_url}/api/v1/contracts/active"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if data.get('code') == '200000' and data.get('data'):
                symbols = [contract['symbol'] for contract in data['data']]
                eth_symbols = [s for s in symbols if 'ETH' in s.upper()]
                
                # Prefer USDT-margined perpetual
                for symbol in eth_symbols:
                    if 'USDT' in symbol and 'M' in symbol:
                        return symbol
                        
                if eth_symbols:
                    return eth_symbols[0]
            
        except Exception as e:
            print(f"⚠️  Could not fetch symbols: {e}")
        
        return "ETHUSDTM"  # Fallback
    
    def get_live_kucoin_data(self):
        """Get live ETH data from KuCoin"""
        try:
            # First ensure we have the correct symbol
            self.symbol = self.get_available_eth_symbols()
            
            # Calculate timestamps for last 2 hours (120 minutes)
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=2)
            
            url = f"{self.base_url}/api/v1/kline/query"
            params = {
                'symbol': self.symbol,
                'granularity': 1,  # 1 minute
                'from': int(start_time.timestamp() * 1000),  # milliseconds
                'to': int(end_time.timestamp() * 1000)       # milliseconds
            }
            
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('code') == '200000' and data.get('data'):
                klines = data['data']
                return klines
            else:
                print(f"❌ KuCoin API error: {data.get('msg', 'Unknown error')}")
                return None
            
        except Exception as e:
            print(f"❌ Error fetching KuCoin data: {e}")
            return None
    
    def process_live_kucoin_data(self, raw_data):
        """Process live KuCoin data"""
        if not raw_data:
            return None
        
        processed_data = []
        for kline in raw_data:
            try:
                # KuCoin kline format: [timestamp, open, close, high, low, volume, turnover]
                processed_data.append({
                    'timestamp': int(kline[0]),  # Already in milliseconds
                    'open': float(kline[1]),
                    'high': float(kline[3]),
                    'low': float(kline[4]),
                    'mark_price': float(kline[2]),  # Use close as mark price
                    'volume': float(kline[5])
                })
            except (IndexError, ValueError, TypeError):
                continue
        
        if not processed_data:
            return None
        
        df = pd.DataFrame(processed_data)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.sort_values('datetime').reset_index(drop=True)
        
        return df
    
    def calculate_live_indicators(self, df):
        """Calculate indicators matching KuCoin training (same as training system)"""
        if len(df) < 25:
            return None
        
        indicators = {}
        price_col = 'mark_price'
        
        # Moving averages (same as training)
        ma_periods = [3, 5, 7, 10, 15, 20]
        for period in ma_periods:
            if len(df) >= period:
                indicators[f'ma{period}'] = df[price_col].rolling(window=period).mean().iloc[-1]
        
        # EMAs (same as training)
        ema_periods = [3, 5, 8, 12, 15, 21]
        for period in ema_periods:
            if len(df) >= period:
                indicators[f'ema{period}'] = df[price_col].ewm(span=period).mean().iloc[-1]
        
        # RSI (same as training)
        if len(df) >= 10:
            delta = df[price_col].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=9).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=9).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            indicators['rsi'] = rsi.iloc[-1]
        
        # MACD (same as training)
        if len(df) >= 17:
            exp1 = df[price_col].ewm(span=8).mean()
            exp2 = df[price_col].ewm(span=17).mean()
            macd = exp1 - exp2
            macd_signal = macd.ewm(span=6).mean()
            
            indicators['macd'] = macd.iloc[-1]
            indicators['macd_signal'] = macd_signal.iloc[-1]
            indicators['macd_histogram'] = (macd - macd_signal).iloc[-1]
        
        # Momentum (same as training)
        for period in [5, 10]:
            if len(df) >= period + 1:
                indicators[f'momentum_{period}'] = df[price_col].iloc[-1] - df[price_col].iloc[-(period+1)]
        
        # Stochastic (same as training)
        if len(df) >= 9:
            low_9 = df['low'].rolling(window=9).min()
            high_9 = df['high'].rolling(window=9).max()
            stoch_k = 100 * ((df[price_col] - low_9) / (high_9 - low_9))
            stoch_d = stoch_k.rolling(window=2).mean()
            
            indicators['stoch_k'] = stoch_k.iloc[-1]
            indicators['stoch_d'] = stoch_d.iloc[-1]
        
        # Williams %R (same as training)
        if len(df) >= 9:
            high_9 = df['high'].rolling(window=9).max()
            low_9 = df['low'].rolling(window=9).min()
            williams_r = -100 * ((high_9 - df[price_col]) / (high_9 - low_9))
            indicators['williams_r'] = williams_r.iloc[-1]
        
        # CCI (same as training)
        if len(df) >= 10:
            tp = (df['high'] + df['low'] + df[price_col]) / 3
            sma_tp = tp.rolling(window=10).mean()
            mad = tp.rolling(window=10).apply(lambda x: np.abs(x - x.mean()).mean())
            cci = (tp - sma_tp) / (0.015 * mad)
            indicators['cci'] = cci.iloc[-1]
        
        # Bollinger Bands (same as training)
        if len(df) >= 15:
            bb_middle = df[price_col].rolling(window=15).mean()
            bb_std = df[price_col].rolling(window=15).std()
            
            indicators['bb_upper'] = (bb_middle + (bb_std * 2)).iloc[-1]
            indicators['bb_middle'] = bb_middle.iloc[-1]
            indicators['bb_lower'] = (bb_middle - (bb_std * 2)).iloc[-1]
        
        # ATR (same as training)
        if len(df) >= 10:
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df[price_col].shift())
            low_close = np.abs(df['low'] - df[price_col].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            atr = true_range.rolling(window=10).mean()
            indicators['atr'] = atr.iloc[-1]
        
        # Current price (same as training)
        indicators['current_mark_price'] = df[price_col].iloc[-1]
        
        # Filter out NaN values
        valid_indicators = {k: v for k, v in indicators.items() if not pd.isna(v)}
        
        return valid_indicators
    
    def find_kucoin_predictions(self, current_indicators):
        """Find predictions from KuCoin patterns"""
        predictions = []
        
        if self.patterns_df is None:
            return predictions
        
        high_confidence_patterns = self.patterns_df[
            (self.patterns_df['combination_size'].isin([3, 4, 5])) &
            (self.patterns_df['occurrences'] >= self.min_occurrences) &
            ((self.patterns_df['rise_percentage'] >= self.min_confidence) |
             (self.patterns_df['fall_percentage'] >= self.min_confidence))
        ]
        
        patterns_checked = 0
        
        for idx, pattern in high_confidence_patterns.iterrows():
            try:
                patterns_checked += 1
                indicator_combo = [ind.strip() for ind in pattern['indicators'].split('+')]
                
                if len(indicator_combo) not in [3, 4, 5]:
                    continue
                
                # Create current ranking
                values = {}
                for indicator in indicator_combo:
                    if indicator in current_indicators and not pd.isna(current_indicators[indicator]):
                        values[indicator] = current_indicators[indicator]
                    else:
                        break
                else:
                    sorted_indicators = sorted(values.items(), key=lambda x: x[1], reverse=True)
                    current_ranking = ' > '.join([item[0] for item in sorted_indicators])
                    
                    if current_ranking == pattern['ranking']:
                        if pattern['rise_percentage'] >= self.min_confidence:
                            pred_type = "UP"
                            confidence = pattern['rise_percentage']
                        else:
                            pred_type = "DOWN"
                            confidence = pattern['fall_percentage']
                        
                        predictions.append({
                            'prediction': pred_type,
                            'confidence': confidence,
                            'combination_size': pattern['combination_size'],
                            'pattern': pattern['ranking'],
                            'indicators': pattern['indicators'],
                            'occurrences': pattern['occurrences']
                        })
                        
            except Exception:
                continue
        
        predictions.sort(key=lambda x: x['confidence'], reverse=True)
        return predictions
    
    def check_previous_prediction_accuracy(self):
        """Check if previous prediction was correct"""
        if self.current_prediction is None:
            return
        
        # Get current price
        raw_data = self.get_live_kucoin_data()
        if not raw_data:
            return
        
        df = self.process_live_kucoin_data(raw_data)
        if df is None or len(df) < 1:
            return
        
        current_price = df.iloc[-1]['mark_price']
        entry_price = self.current_prediction['entry_price']
        predicted_direction = self.current_prediction['prediction']
        
        # Check if prediction was correct
        if predicted_direction == "UP":
            is_correct = current_price > entry_price
        else:  # DOWN
            is_correct = current_price < entry_price
        
        # Update counters
        self.total_predictions += 1
        if is_correct:
            self.correct_predictions += 1
            result = "✅ CORRECT"
        else:
            self.wrong_predictions += 1
            result = "❌ WRONG"
        
        # Calculate price change
        price_change_pct = ((current_price - entry_price) / entry_price) * 100
        
        print(f"\n📊 PREVIOUS PREDICTION RESULT:")
        print(f"   Predicted: {predicted_direction} | Actual: {'UP' if current_price > entry_price else 'DOWN'}")
        print(f"   Price: ${entry_price:.2f} → ${current_price:.2f} ({price_change_pct:+.2f}%)")
        print(f"   Result: {result}")
        print(f"   Confidence was: {self.current_prediction['confidence']:.1f}%")
        
        # Store in history
        self.prediction_history.append({
            'timestamp': self.current_prediction['timestamp'],
            'prediction': predicted_direction,
            'confidence': self.current_prediction['confidence'],
            'entry_price': entry_price,
            'exit_price': current_price,
            'price_change_pct': price_change_pct,
            'is_correct': is_correct,
            'result': result,
            'pattern': self.current_prediction['pattern']
        })
        
        print("-" * 50)
    
    def make_new_prediction(self):
        """Make a new prediction"""
        # Get live data
        raw_data = self.get_live_kucoin_data()
        if not raw_data:
            print("⚠️ Failed to get KuCoin data")
            return
        
        # Process data
        df = self.process_live_kucoin_data(raw_data)
        if df is None or len(df) < 25:
            print("⚠️ Insufficient KuCoin data for indicators")
            return
        
        # Calculate indicators
        indicators = self.calculate_live_indicators(df)
        if not indicators:
            print("⚠️ Could not calculate indicators")
            return
        
        current_price = indicators['current_mark_price']
        
        # Find predictions
        predictions = self.find_kucoin_predictions(indicators)
        
        if predictions:
            best_prediction = predictions[0]
            pred_type = best_prediction['prediction']
            
            # Store current prediction
            self.current_prediction = {
                'timestamp': datetime.now(),
                'prediction': pred_type,
                'confidence': best_prediction['confidence'],
                'entry_price': current_price,
                'pattern': best_prediction['pattern'],
                'indicators': best_prediction['indicators'],
                'occurrences': best_prediction['occurrences']
            }
            
            print(f"\n🚨 NEW KUCOIN PREDICTION!")
            print(f"📈 Direction: {pred_type}")
            print(f"🎯 Confidence: {best_prediction['confidence']:.1f}%")
            print(f"💰 Entry Price: ${current_price:.2f}")
            print(f"📊 Pattern Size: {best_prediction['combination_size']} indicators")
            print(f"🔧 Pattern: {best_prediction['pattern']}")
            print(f"📈 Historical Reliability: {best_prediction['occurrences']} occurrences")
            print(f"🏢 Source: KuCoin Training Data")
        else:
            print(f"\n📊 No high-confidence patterns found")
            print(f"💰 Current ETH Price: ${current_price:.2f}")
            self.current_prediction = None
    
    def display_current_stats(self):
        """Display current prediction statistics"""
        accuracy = (self.correct_predictions / self.total_predictions * 100) if self.total_predictions > 0 else 0
        
        print(f"\n📊 CURRENT PREDICTION ACCURACY STATS:")
        print(f"   Total Predictions: {self.total_predictions}")
        print(f"   Correct Predictions: {self.correct_predictions}")
        print(f"   Wrong Predictions: {self.wrong_predictions}")
        print(f"   Accuracy Rate: {accuracy:.1f}%")
        
        if self.current_prediction:
            print(f"\n🎯 ACTIVE PREDICTION:")
            print(f"   Direction: {self.current_prediction['prediction']}")
            print(f"   Confidence: {self.current_prediction['confidence']:.1f}%")
            print(f"   Entry Price: ${self.current_prediction['entry_price']:.2f}")
            print(f"   Made at: {self.current_prediction['timestamp'].strftime('%H:%M:%S')}")
        else:
            print(f"\n🎯 ACTIVE PREDICTION: None")
    
    def run_prediction_tracker(self):
        """Run the prediction accuracy tracker"""
        print("🎯 ETH KUCOIN PREDICTION ACCURACY TRACKER")
        print("="*60)
        print(f"🏢 Exchange: KuCoin Futures API")
        print(f"📊 Focus: Prediction accuracy tracking only")
        print(f"⏰ Update Frequency: Once per minute")
        print(f"🎯 Confidence Threshold: {self.min_confidence}%+")
        print(f"📈 Symbol: {self.symbol}")
        print("="*60)
        
        first_run = True
        
        while True:
            try:
                current_time = datetime.now()
                print(f"\n⏰ Update at {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Check previous prediction accuracy (skip on first run)
                if not first_run:
                    self.check_previous_prediction_accuracy()
                else:
                    first_run = False
                
                # Make new prediction
                self.make_new_prediction()
                
                # Display current statistics
                self.display_current_stats()
                
                # Save results periodically
                if self.prediction_history:
                    self.save_prediction_results()
                
                print(f"\n⏳ Waiting for next minute...")
                print("=" * 60)
                
                # Wait until next minute
                time.sleep(60)
                
            except KeyboardInterrupt:
                print("\n🛑 Prediction tracker stopped")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                time.sleep(60)
        
        # Final statistics
        self.display_final_results()
    
    def save_prediction_results(self):
        """Save prediction results to CSV"""
        if self.prediction_history:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            df = pd.DataFrame(self.prediction_history)
            filename = f"eth_kucoin_prediction_accuracy_{timestamp}.csv"
            df.to_csv(filename, index=False)
            print(f"💾 Results saved: {filename}")
    
    def display_final_results(self):
        """Display final prediction accuracy results"""
        print(f"\n🏆 FINAL PREDICTION ACCURACY RESULTS")
        print("="*60)
        
        if self.total_predictions == 0:
            print("No predictions completed")
            return
        
        accuracy = (self.correct_predictions / self.total_predictions) * 100
        
        print(f"📊 OVERALL STATISTICS:")
        print(f"   Total Predictions Made: {self.total_predictions}")
        print(f"   Correct Predictions: {self.correct_predictions}")
        print(f"   Wrong Predictions: {self.wrong_predictions}")
        print(f"   Overall Accuracy: {accuracy:.1f}%")
        
        if self.prediction_history:
            df = pd.DataFrame(self.prediction_history)
            
            # UP vs DOWN accuracy
            up_predictions = df[df['prediction'] == 'UP']
            down_predictions = df[df['prediction'] == 'DOWN']
            
            if len(up_predictions) > 0:
                up_accuracy = (up_predictions['is_correct'].sum() / len(up_predictions)) * 100
                print(f"   UP Predictions Accuracy: {up_accuracy:.1f}% ({up_predictions['is_correct'].sum()}/{len(up_predictions)})")
            
            if len(down_predictions) > 0:
                down_accuracy = (down_predictions['is_correct'].sum() / len(down_predictions)) * 100
                print(f"   DOWN Predictions Accuracy: {down_accuracy:.1f}% ({down_predictions['is_correct'].sum()}/{len(down_predictions)})")
            
            # Average confidence of correct vs wrong predictions
            correct_preds = df[df['is_correct'] == True]
            wrong_preds = df[df['is_correct'] == False]
            
            if len(correct_preds) > 0:
                avg_correct_confidence = correct_preds['confidence'].mean()
                print(f"   Average Confidence (Correct): {avg_correct_confidence:.1f}%")
            
            if len(wrong_preds) > 0:
                avg_wrong_confidence = wrong_preds['confidence'].mean()
                print(f"   Average Confidence (Wrong): {avg_wrong_confidence:.1f}%")
            
            # Save final results
            self.save_prediction_results()

def main():
    print("🚀 ETH KUCOIN PREDICTION ACCURACY TRACKER")
    print("="*60)
    print("🎯 Features:")
    print("   - Pure prediction accuracy tracking")
    print("   - No trading simulation")
    print("   - Updates once per minute")
    print("   - Tracks total predictions, wins, accuracy")
    print("   - Uses KuCoin Futures API (Cloud friendly)")
    print("="*60)
    
    tracker = ETHKuCoinPredictionAccuracyTracker('eth_kucoin_patterns_parallel_3_4_5.csv')
    
    print("\nSelect mode:")
    print("1. Run prediction accuracy tracker (continuous)")
    print("2. Single prediction check")
    
    try:
        choice = input("\nEnter choice (1-2): ").strip()
        
        if choice == '1':
            tracker.run_prediction_tracker()
            
        elif choice == '2':
            print("\n🔍 Single KuCoin prediction check...")
            tracker.make_new_prediction()
            tracker.display_current_stats()
            
        else:
            print("Invalid choice")
            
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
