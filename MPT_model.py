# MPT &  Fama-French 5팩터 혼합 모델

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy.stats import norm
from datetime import datetime, timedelta
from sklearn.covariance import ledoit_wolf, empirical_covariance

class PortfolioOptimizer:
    def __init__(self):
        self.raw_data = None
        self.processed_data = None
        self.stock_returns = None
        self.factors = None
        self.selected_stocks = None
        self.risk_preference = None
        self.initial_investment = None
        self.initial_weights = None
        self.constraints = None
        self.factor_betas = None
        self.factor_returns = None
        self.expected_returns = None
        self.covariance_matrix = None
        self.efficient_frontier = None
        self.optimal_portfolio = None
        self.ml_model = RandomForestRegressor(n_estimators=100, random_state=42)

    def collect_data(self, start_date, end_date):
        """
        여러 API에서 데이터를 수집하고 통합하는 메소드
        """
        self.start_date = start_date
        self.end_date = end_date

        # 주식 가격 및 거래량 데이터 수집
        self.stock_data = self._get_stock_data()

        # 재무제표 데이터 수집
        self.financial_data = self._get_financial_data()

        # 시장 지표 데이터 수집
        self.market_data = self._get_market_data()

        # Fama-French 팩터 데이터 수집
        self.ff_factors = self._get_fama_french_factors()

        # 경제 지표 데이터 수집
        self.economic_indicators = self._get_economic_indicators()

        # 산업 분류 데이터 수집
        self.industry_classification = self._get_industry_classification()

        # 기업 이벤트 데이터 수집
        self.corporate_events = self._get_corporate_events()

        # 시장 센티먼트 데이터 수집
        self.market_sentiment = self._get_market_sentiment()

        # 옵션 시장 데이터 수집
        self.options_data = self._get_options_data()

        # 환율 데이터 수집
        self.exchange_rates = self._get_exchange_rates()

        # 데이터 통합 및 전처리
        self._integrate_and_preprocess_data()

    def _get_stock_data(self):
        # 한국 주식 데이터 (KRX API 사용 예시)
        krx_data = self._fetch_krx_data(self.start_date, self.end_date)

        # 미국 주식 데이터 (Alpha Vantage API 사용 예시)
        us_data = self._fetch_alpha_vantage_data(self.start_date, self.end_date)

        # 데이터 통합
        return pd.concat([krx_data, us_data])

    def _get_financial_data(self):
        # 한국 기업 재무제표 (OpenDART API 사용 예시)
        kr_financial = self._fetch_opendart_data()

        # 미국 기업 재무제표 (SEC EDGAR API 사용 예시)
        us_financial = self._fetch_sec_edgar_data()

        # 데이터 통합
        return pd.concat([kr_financial, us_financial])

    def _get_market_data(self):
        # 한국 시장 지표 (ECOS API 사용 예시)
        kr_market = self._fetch_ecos_data()

        # 미국 시장 지표 (FRED API 사용 예시)
        us_market = self._fetch_fred_data()

        # 데이터 통합
        return pd.concat([kr_market, us_market])

    def _get_fama_french_factors(self):
        # Kenneth French's Data Library에서 데이터 다운로드 및 처리
        return self._fetch_fama_french_data()

    def _get_economic_indicators(self):
        # 국제 경제 지표 (World Bank API 사용 예시)
        return self._fetch_world_bank_data()

    def _get_industry_classification(self):
        # GICS 데이터 또는 주식 정보 API에서 산업 분류 정보 추출
        return self._fetch_industry_classification_data()

    def _get_corporate_events(self):
        # 한국 기업 이벤트 (OpenDART API 사용 예시)
        kr_events = self._fetch_opendart_events()

        # 미국 기업 이벤트 (SEC EDGAR API 사용 예시)
        us_events = self._fetch_sec_edgar_events()

        # 데이터 통합
        return pd.concat([kr_events, us_events])

    def _get_market_sentiment(self):
        # 뉴스 데이터 (NewsAPI 사용 예시)
        news_sentiment = self._fetch_news_api_data()

        # 소셜 미디어 데이터 (Twitter API 사용 예시)
        social_sentiment = self._fetch_twitter_data()

        # 데이터 통합 및 감성 분석
        return self._analyze_sentiment(news_sentiment, social_sentiment)

    def _get_options_data(self):
        # 한국 옵션 데이터 (KRX 파생상품 시장 데이터 API 사용 예시)
        kr_options = self._fetch_krx_options_data()

        # 미국 옵션 데이터 (TD Ameritrade API 사용 예시)
        us_options = self._fetch_td_ameritrade_options_data()

        # 데이터 통합
        return pd.concat([kr_options, us_options])

    def _get_exchange_rates(self):
        # 환율 데이터 (Open Exchange Rates API 사용 예시)
        return self._fetch_exchange_rates_data()

    def _integrate_and_preprocess_data(self):
        # 모든 수집된 데이터를 통합하고 전처리하는 로직
        # 예: 날짜 정렬, 결측치 처리, 이상치 제거 등
        pass

    # 각 API에 대한 실제 데이터 fetch 함수들
    # 실제 구현 시에는 각 API의 요구사항에 맞게 구현해야 함
    def _fetch_krx_data(self, start_date, end_date):
        # KRX API 호출 및 데이터 처리 로직
        pass

    def _fetch_alpha_vantage_data(self, start_date, end_date):
        # Alpha Vantage API 호출 및 데이터 처리 로직
        pass

    def manage_data_quality(self):
        """
        수집된 모든 데이터에 대해 품질 관리 작업을 수행합니다.
        """
        # 주식 가격 및 거래량 데이터 품질 관리
        self.stock_data = self._manage_stock_data_quality(self.stock_data)

        # 재무제표 데이터 품질 관리
        self.financial_data = self._manage_financial_data_quality(self.financial_data)

        # 시장 지표 데이터 품질 관리
        self.market_data = self._manage_market_data_quality(self.market_data)

        # Fama-French 팩터 데이터 품질 관리
        self.ff_factors = self._manage_ff_factors_quality(self.ff_factors)

        # 경제 지표 데이터 품질 관리
        self.economic_indicators = self._manage_economic_indicators_quality(self.economic_indicators)

        # 산업 분류 데이터 품질 관리
        self.industry_classification = self._manage_industry_classification_quality(self.industry_classification)

        # 기업 이벤트 데이터 품질 관리
        self.corporate_events = self._manage_corporate_events_quality(self.corporate_events)

        # 시장 센티먼트 데이터 품질 관리
        self.market_sentiment = self._manage_market_sentiment_quality(self.market_sentiment)

        # 옵션 시장 데이터 품질 관리
        self.options_data = self._manage_options_data_quality(self.options_data)

        # 환율 데이터 품질 관리
        self.exchange_rates = self._manage_exchange_rates_quality(self.exchange_rates)

        # 전체 데이터 정합성 확인
        self._check_data_consistency()

    def _manage_stock_data_quality(self, data):
        # 결측치 처리
        data = data.fillna(method='ffill').fillna(method='bfill')

        # 이상치 처리 (Z-score 방법)
        z_scores = np.abs(stats.zscore(data))
        data = data[(z_scores < 3).all(axis=1)]

        # 거래량이 0인 데이터 제거
        data = data[data['volume'] > 0]

        # 주가 분할 및 배당 조정
        data = self._adjust_for_splits_and_dividends(data)

        return data

    def _manage_financial_data_quality(self, data):
        # 결측치 처리
        data = data.fillna(data.mean())

        # 이상치 처리 (IQR 방법)
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        data = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]

        # 재무비율 계산 및 이상치 제거
        data['debt_to_equity'] = data['total_debt'] / data['total_equity']
        data = data[(data['debt_to_equity'] > 0) & (data['debt_to_equity'] < 10)]

        return data

    def _manage_market_data_quality(self, data):
        # 결측치 처리
        data = data.interpolate()

        # 이상치 처리 (롤링 평균 사용)
        rolling_mean = data.rolling(window=30).mean()
        rolling_std = data.rolling(window=30).std()
        data = data[(data > (rolling_mean - 3 * rolling_std)) & (data < (rolling_mean + 3 * rolling_std))]

        return data

    def _manage_ff_factors_quality(self, data):
        # 결측치 처리
        data = data.fillna(data.mean())

        # 이상치 처리 (Winsorization)
        for column in data.columns:
            data[column] = stats.mstats.winsorize(data[column], limits=[0.01, 0.01])

        return data

    def _manage_economic_indicators_quality(self, data):
        # 결측치 처리
        data = data.interpolate(method='time')

        # 계절성 조정 (필요한 경우)
        # 예: data = self._adjust_seasonality(data)

        return data

    def _manage_industry_classification_quality(self, data):
        # 결측치 처리 (최빈값으로 대체)
        data = data.fillna(data.mode().iloc[0])

        # 일관성 확인
        # 예: 동일 기업의 산업 분류가 시간에 따라 급격히 변하지 않는지 확인

        return data

    def _manage_corporate_events_quality(self, data):
        # 중복 이벤트 제거
        data = data.drop_duplicates()

        # 이벤트 날짜 검증
        data = data[data['event_date'] <= pd.Timestamp.now()]

        return data

    def _manage_market_sentiment_quality(self, data):
        # 결측치 처리
        data = data.fillna(0)  # 중립적 감성으로 가정

        # 이상치 처리 (감성 점수가 -1에서 1 사이인지 확인)
        data['sentiment_score'] = data['sentiment_score'].clip(-1, 1)

        return data

    def _manage_options_data_quality(self, data):
        # 결측치 처리
        data = data.fillna(method='ffill')

        # 이상치 처리 (비현실적인 변동성 제거)
        data = data[data['implied_volatility'] < 5]  # 500% 이상의 변동성 제거

        return data

    def _manage_exchange_rates_quality(self, data):
        # 결측치 처리
        data = data.interpolate()

        # 이상치 처리 (급격한 변동 탐지)
        pct_change = data.pct_change()
        data = data[np.abs(pct_change) < 0.1]  # 10% 이상 급변하는 환율 제거

        return data

    def _check_data_consistency(self):
        # 모든 데이터의 날짜 범위 일치 확인
        date_ranges = {
            'stock_data': (self.stock_data.index.min(), self.stock_data.index.max()),
            'market_data': (self.market_data.index.min(), self.market_data.index.max()),
            'ff_factors': (self.ff_factors.index.min(), self.ff_factors.index.max()),
            'exchange_rates': (self.exchange_rates.index.min(), self.exchange_rates.index.max()),
        }

        if len(set(date_ranges.values())) > 1:
            print("Warning: Date ranges are not consistent across all datasets.")

        # 기업 코드 일치 확인
        stock_codes = set(self.stock_data['code'])
        financial_codes = set(self.financial_data['code'])
        if stock_codes != financial_codes:
            print("Warning: Mismatch in company codes between stock data and financial data.")

    def _adjust_for_splits_and_dividends(self, data):
        # 주식 분할 및 배당 정보를 이용하여 주가 조정
        # 실제 구현 시 기업 이벤트 데이터를 활용하여 조정
        return data

    def construct_fama_french_factors(self):
        """
        Fama-French 5팩터 모델 구성
        - 시장(Rm-Rf), 규모(SMB), 가치(HML), 수익성(RMW), 투자(CMA) 팩터 계산
        """
        # 시장 팩터 (Rm-Rf) 계산
        market_return = self.stock_data['market_index'].pct_change()
        risk_free_rate = self.market_data['risk_free_rate']
        market_factor = market_return - risk_free_rate

        # 기업 특성 계산
        self.stock_data['market_cap'] = self.stock_data['price'] * self.stock_data['shares_outstanding']
        self.stock_data['book_to_market'] = self.financial_data['book_value'] / self.stock_data['market_cap']
        self.stock_data['operating_profitability'] = self.financial_data['operating_income'] / self.financial_data['book_value']
        self.stock_data['investment'] = (self.financial_data['total_assets'] - self.financial_data['total_assets'].shift(1)) / self.financial_data['total_assets'].shift(1)

        # 포트폴리오 구성을 위한 분위수 계산
        market_cap_median = self.stock_data['market_cap'].median()
        btm_30th = self.stock_data['book_to_market'].quantile(0.3)
        btm_70th = self.stock_data['book_to_market'].quantile(0.7)
        op_median = self.stock_data['operating_profitability'].median()
        inv_median = self.stock_data['investment'].median()

        # SMB (Small Minus Big) 팩터 계산
        small_return = self.stock_data[self.stock_data['market_cap'] <= market_cap_median]['return'].mean()
        big_return = self.stock_data[self.stock_data['market_cap'] > market_cap_median]['return'].mean()
        smb_factor = small_return - big_return

        # HML (High Minus Low) 팩터 계산
        high_btm_return = self.stock_data[self.stock_data['book_to_market'] >= btm_70th]['return'].mean()
        low_btm_return = self.stock_data[self.stock_data['book_to_market'] <= btm_30th]['return'].mean()
        hml_factor = high_btm_return - low_btm_return

        # RMW (Robust Minus Weak) 팩터 계산
        robust_return = self.stock_data[self.stock_data['operating_profitability'] > op_median]['return'].mean()
        weak_return = self.stock_data[self.stock_data['operating_profitability'] <= op_median]['return'].mean()
        rmw_factor = robust_return - weak_return

        # CMA (Conservative Minus Aggressive) 팩터 계산
        conservative_return = self.stock_data[self.stock_data['investment'] <= inv_median]['return'].mean()
        aggressive_return = self.stock_data[self.stock_data['investment'] > inv_median]['return'].mean()
        cma_factor = conservative_return - aggressive_return

        # 팩터 데이터프레임 생성
        self.factors = pd.DataFrame({
            'Rm-Rf': market_factor,
            'SMB': smb_factor,
            'HML': hml_factor,
            'RMW': rmw_factor,
            'CMA': cma_factor
        })

        # 팩터 간 상관관계 분석
        factor_correlation = self.factors.corr()
        print("Fama-French 5팩터 상관관계:")
        print(factor_correlation)

        # 팩터 수익률 통계 출력
        factor_stats = self.factors.describe()
        print("\nFama-French 5팩터 통계:")
        print(factor_stats)

    def estimate_factor_betas(self):
        """
        머신러닝 모델(Random Forest)을 사용하여 선택된 주식의 팩터 베타 추정
        """
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, r2_score

        # 데이터 준비
        X = self.factors
        y = self.stock_returns

        # 학습 데이터와 테스트 데이터 분리
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 모델 학습
        self.ml_model.fit(X_train, y_train)

        # 테스트 데이터로 예측
        y_pred = self.ml_model.predict(X_test)

        # 모델 성능 평가
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"모델 성능 - MSE: {mse:.4f}, R2: {r2:.4f}")

        # 팩터 베타 추정 (Feature Importance를 베타로 사용)
        self.factor_betas = pd.DataFrame(
            self.ml_model.feature_importances_,
            index=self.factors.columns,
            columns=['Beta']
        )

        print("\n추정된 팩터 베타:")
        print(self.factor_betas)

        # 팩터 베타의 안정성 검증
        self.validate_factor_betas()

    def validate_factor_betas(self):
        """
        추정된 팩터 베타의 안정성을 검증
        """
        from sklearn.model_selection import cross_val_score

        # 교차 검증 수행
        cv_scores = cross_val_score(self.ml_model, self.factors, self.stock_returns, cv=5)

        print("\n교차 검증 결과:")
        print(f"평균 R2: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        # 부트스트래핑을 통한 베타 신뢰구간 추정
        n_iterations = 1000
        n_samples = len(self.factors)

        bootstrap_betas = []
        for _ in range(n_iterations):
            indices = np.random.randint(0, n_samples, n_samples)
            X_bootstrap = self.factors.iloc[indices]
            y_bootstrap = self.stock_returns.iloc[indices]

            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_bootstrap, y_bootstrap)
            bootstrap_betas.append(model.feature_importances_)

        bootstrap_betas = np.array(bootstrap_betas)

        # 95% 신뢰구간 계산
        confidence_intervals = np.percentile(bootstrap_betas, [2.5, 97.5], axis=0)

        print("\n팩터 베타 95% 신뢰구간:")
        for i, factor in enumerate(self.factors.columns):
            print(f"{factor}: ({confidence_intervals[0][i]:.4f}, {confidence_intervals[1][i]:.4f})")

    def estimate_factor_returns(self):
        """
        머신러닝 모델을 사용하여 Fama-French 5팩터의 기대수익률 추정
        """
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, r2_score
        from sklearn.preprocessing import StandardScaler
        from statsmodels.tsa.arima.model import ARIMA
        import warnings

        warnings.filterwarnings('ignore')  # ARIMA 모델의 경고 메시지 억제

        # 데이터 준비
        factor_returns = self.factors.pct_change().dropna()

        # 각 팩터별로 기대수익률 추정
        self.factor_returns = {}
        for factor in factor_returns.columns:
            print(f"\n{factor} 팩터 기대수익률 추정:")

            # 1. 시계열 모델 (ARIMA) 사용
            model = ARIMA(factor_returns[factor], order=(1,1,1))
            results = model.fit()
            arima_forecast = results.forecast(steps=1)[0]

            print(f"ARIMA 모델 예측: {arima_forecast:.4f}")

            # 2. 머신러닝 모델 (Random Forest) 사용
            X = factor_returns.drop(columns=[factor])
            y = factor_returns[factor]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_train_scaled, y_train)

            y_pred = rf_model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            print(f"Random Forest 모델 성능 - MSE: {mse:.4f}, R2: {r2:.4f}")

            rf_forecast = rf_model.predict(scaler.transform(X.iloc[-1:]))[0]
            print(f"Random Forest 모델 예측: {rf_forecast:.4f}")

            # 3. 앙상블 방법: ARIMA와 Random Forest 예측의 가중 평균
            ensemble_forecast = 0.5 * arima_forecast + 0.5 * rf_forecast
            print(f"앙상블 예측: {ensemble_forecast:.4f}")

            # 최종 기대수익률 저장
            self.factor_returns[factor] = ensemble_forecast

        # 팩터 기대수익률 요약
        print("\n추정된 팩터 기대수익률:")
        for factor, return_value in self.factor_returns.items():
            print(f"{factor}: {return_value:.4f}")

        # 기대수익률의 신뢰성 검증
        self.validate_factor_returns(factor_returns)

    def validate_factor_returns(self, factor_returns):
        """
        추정된 팩터 기대수익률의 신뢰성을 검증
        """
        from scipy import stats

        print("\n팩터 기대수익률 신뢰성 검증:")

        for factor in factor_returns.columns:
            # 과거 수익률의 평균 및 표준편차 계산
            historical_mean = factor_returns[factor].mean()
            historical_std = factor_returns[factor].std()

            # 추정된 기대수익률
            estimated_return = self.factor_returns[factor]

            # z-점수 계산
            z_score = (estimated_return - historical_mean) / historical_std

            # 신뢰구간 계산 (95% 신뢰수준)
            confidence_interval = stats.norm.interval(0.95, loc=historical_mean, scale=historical_std/np.sqrt(len(factor_returns)))

            print(f"\n{factor}:")
            print(f"  과거 평균 수익률: {historical_mean:.4f}")
            print(f"  추정 기대수익률: {estimated_return:.4f}")
            print(f"  z-점수: {z_score:.4f}")
            print(f"  95% 신뢰구간: ({confidence_interval[0]:.4f}, {confidence_interval[1]:.4f})")

            if confidence_interval[0] <= estimated_return <= confidence_interval[1]:
                print("  결과: 추정된 기대수익률이 95% 신뢰구간 내에 있습니다.")
            else:
                print("  결과: 주의! 추정된 기대수익률이 95% 신뢰구간을 벗어났습니다.")

    def calculate_expected_returns(self):
        """
        추정된 베타와 팩터 수익률을 사용하여 각 주식의 기대수익률 계산
        """
        # 팩터 수익률을 numpy 배열로 변환
        factor_returns = np.array([self.factor_returns[factor] for factor in self.factors.columns])

        # 각 주식의 기대수익률 계산 (베타와 팩터 수익률의 행렬 곱)
        self.expected_returns = np.dot(self.factor_betas, factor_returns)

        # 결과를 pandas Series로 변환하여 주식 이름을 인덱스로 사용
        self.expected_returns = pd.Series(self.expected_returns, index=self.selected_stocks)

        print("\n각 주식의 기대수익률:")
        for stock, expected_return in self.expected_returns.items():
            print(f"{stock}: {expected_return:.4f}")

        # 기대수익률의 합리성 검증
        self.validate_expected_returns()

    def validate_expected_returns(self):
        """
        계산된 기대수익률의 합리성을 검증
        """
        print("\n기대수익률 합리성 검증:")

        # 1. 범위 확인
        min_return = self.expected_returns.min()
        max_return = self.expected_returns.max()
        print(f"기대수익률 범위: {min_return:.4f} ~ {max_return:.4f}")

        if min_return < -0.5 or max_return > 0.5:
            print("주의: 일부 기대수익률이 비정상적으로 높거나 낮습니다.")

        # 2. 과거 수익률과 비교
        historical_mean = self.stock_returns.mean()
        correlation = self.expected_returns.corr(historical_mean)
        print(f"과거 평균 수익률과의 상관관계: {correlation:.4f}")

        if correlation < 0.3:
            print("주의: 기대수익률이 과거 수익률과 낮은 상관관계를 보입니다.")

        # 3. 산업 평균과 비교 (예시 - 실제 구현 시 산업 분류 데이터 필요)
        industry_average = self.expected_returns.mean()
        outliers = self.expected_returns[abs(self.expected_returns - industry_average) > 2 * self.expected_returns.std()]

        if not outliers.empty:
            print("\n산업 평균에서 크게 벗어난 주식:")
            for stock, value in outliers.items():
                print(f"{stock}: {value:.4f}")

    def construct_covariance_matrix(self):
        """
        선택된 주식의 수익률 데이터를 사용하여 공분산 행렬 구성
        """

        # 표본 공분산 행렬 계산
        sample_cov = empirical_covariance(self.stock_returns)

        # Ledoit-Wolf shrinkage 방법을 사용한 공분산 행렬 계산
        lw_cov, shrinkage = ledoit_wolf(self.stock_returns)

        # 두 방법의 조건수 비교
        cond_sample = np.linalg.cond(sample_cov)
        cond_lw = np.linalg.cond(lw_cov)

        print(f"표본 공분산 행렬 조건수: {cond_sample:.2f}")
        print(f"Ledoit-Wolf 축소 후 조건수: {cond_lw:.2f}")
        print(f"Ledoit-Wolf 축소 강도: {shrinkage:.4f}")

        # 조건수가 더 낮은 (더 안정적인) 방법 선택
        if cond_lw < cond_sample:
            print("Ledoit-Wolf 방법이 선택되었습니다.")
            self.covariance_matrix = lw_cov
        else:
            print("표본 공분산 행렬이 선택되었습니다.")
            self.covariance_matrix = sample_cov

        # 공분산 행렬을 DataFrame으로 변환 (시각화 및 분석을 위해)
        self.covariance_matrix = pd.DataFrame(self.covariance_matrix,
                                              index=self.selected_stocks,
                                              columns=self.selected_stocks)

        # 상관관계 행렬 계산
        correlation_matrix = self.stock_returns.corr()

        # 높은 상관관계 (절대값 0.8 이상) 확인
        high_corr = np.abs(correlation_matrix) > 0.8
        np.fill_diagonal(high_corr, False)
        if np.any(high_corr):
            print("\n주의: 다음 주식 쌍 간 높은 상관관계가 발견되었습니다:")
            for i, j in zip(*np.where(high_corr)):
                print(f"{correlation_matrix.index[i]} - {correlation_matrix.columns[j]}: {correlation_matrix.iloc[i, j]:.2f}")

        # 공분산 행렬 시각화
        plt.figure(figsize=(12, 10))
        sns.heatmap(self.covariance_matrix, annot=True, cmap='coolwarm', fmt='.2e')
        plt.title('공분산 행렬 히트맵')
        plt.tight_layout()
        plt.show()

        # 상관관계 행렬 시각화
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
        plt.title('상관관계 행렬 히트맵')
        plt.tight_layout()
        plt.show()

# 여기부터 수정------------------------------------------------------------------------------------------------

    def calculate_transaction_costs(self, current_portfolio, new_portfolio):
        """
        거래 비용 모델: 포트폴리오 재조정에 따른 거래 비용 계산
        """
        # 간단한 예시로 거래량의 0.1%를 거래 비용으로 가정
        if current_portfolio is None:
            return 0.001 * np.sum(np.abs(new_portfolio))
        return 0.001 * np.sum(np.abs(new_portfolio - current_portfolio))

    def optimize_portfolio(self):
        """
        MPT 최적화 수행
        - 목적 함수: 효용 최대화 (U = E(R) - 0.5 * λ * σ^2 - 거래비용)
        - 제약 조건: 투자 비중 합 = 1, 개별 주식 최대 투자 비중 등
        """
        n = len(self.selected_stocks)

        def objective(weights):
            portfolio_return = np.sum(self.expected_returns * weights)
            portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(self.covariance_matrix, weights)))
            transaction_costs = self.calculate_transaction_costs(self.initial_weights, weights)
            utility = portfolio_return - 0.5 * self.risk_preference * portfolio_risk**2 - transaction_costs
            return -utility  # 최소화 문제로 변환

        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # 투자 비중 합 = 1
        ]

        # 투자 제약 조건 추가
        if self.constraints:
            # 예: 특정 섹터 최대 비중 제한
            constraints.append({'type': 'ineq', 'fun': lambda x: 0.5 - np.sum(x[:len(self.constraints)])})

        bounds = [(0, 0.5) for _ in range(n)]  # 개별 주식 최대 투자 비중 50%

        result = minimize(objective, self.initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

        self.optimal_portfolio = result.x

    def calculate_efficient_frontier(self):
        """
        다양한 위험 수준에 대한 최적 포트폴리오를 계산하여 효율적 프론티어 도출
        """
        n_points = 100
        risk_levels = np.linspace(0.1, 0.5, n_points)
        returns = []
        risks = []

        for risk in risk_levels:
            risk_preference_temp = self.risk_preference
            self.risk_preference = 1 / risk
            self.optimize_portfolio()
            returns.append(np.sum(self.expected_returns * self.optimal_portfolio))
            risks.append(np.sqrt(np.dot(self.optimal_portfolio.T, np.dot(self.covariance_matrix, self.optimal_portfolio))))
            self.risk_preference = risk_preference_temp

        self.efficient_frontier = pd.DataFrame({'Return': returns, 'Risk': risks})

    def calculate_risk_metrics(self, portfolio):
        """
        포트폴리오의 다양한 위험 지표 계산 (VaR, CVaR 등)
        """
        portfolio_returns = np.dot(self.stock_returns, portfolio)

        var_95 = np.percentile(portfolio_returns, 5)
        cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()

        return {
            'VaR_95': -var_95,
            'CVaR_95': -cvar_95,
            'Volatility': portfolio_returns.std()
        }

    def select_optimal_portfolio(self):
        """
        투자자의 위험 선호도와 다양한 위험 지표를 고려하여 효율적 프론티어에서 최적 포트폴리오 선택
        """
        # 샤프 비율을 최대화하는 포트폴리오 선택
        risk_free_rate = 0.02  # 예시 무위험 수익률
        self.efficient_frontier['Sharpe'] = (self.efficient_frontier['Return'] - risk_free_rate) / self.efficient_frontier['Risk']
        optimal_portfolio = self.efficient_frontier.loc[self.efficient_frontier['Sharpe'].idxmax()]

        # 위험 선호도를 고려한 조정
        risk_adjusted_portfolio = self.efficient_frontier.iloc[int(len(self.efficient_frontier) * self.risk_preference / 10)]

        # 위험 지표를 고려한 최종 선택
        risk_metrics_optimal = self.calculate_risk_metrics(optimal_portfolio)
        risk_metrics_adjusted = self.calculate_risk_metrics(risk_adjusted_portfolio)

        if risk_metrics_adjusted['VaR_95'] < risk_metrics_optimal['VaR_95']:
            self.optimal_portfolio = risk_adjusted_portfolio
        else:
            self.optimal_portfolio = optimal_portfolio

    def rebalance_portfolio(self):
        """
        현재 포트폴리오를 새로운 최적 포트폴리오로 재조정, 거래 비용 고려
        """
        transaction_costs = self.calculate_transaction_costs(self.initial_weights, self.optimal_portfolio)
        self.optimal_portfolio -= transaction_costs / len(self.optimal_portfolio)  # 거래 비용을 균등하게 차감
        self.optimal_portfolio /= np.sum(self.optimal_portfolio)  # 정규화

    def run_scenario_analysis(self, num_scenarios=1000):
        """
        다양한 시장 상황을 가정한 시나리오 분석 수행
        """
        mean_returns = self.stock_returns.mean()
        cov_matrix = self.stock_returns.cov()

        simulated_returns = np.random.multivariate_normal(mean_returns, cov_matrix, num_scenarios)
        portfolio_returns = np.dot(simulated_returns, self.optimal_portfolio)

        return portfolio_returns

    def evaluate_portfolio_robustness(self, scenario_results):
        """
        시나리오 분석 결과를 바탕으로 포트폴리오 견고성 평가
        """
        return {
            'Mean Return': np.mean(scenario_results),
            'Std Dev': np.std(scenario_results),
            'VaR_95': -np.percentile(scenario_results, 5),
            'CVaR_95': -np.mean(scenario_results[scenario_results <= np.percentile(scenario_results, 5)])
        }

    def backtest_portfolio(self):
        """
        과거 데이터를 사용하여 포트폴리오 성과 시뮬레이션
        """
        portfolio_returns = np.dot(self.stock_returns, self.optimal_portfolio)

        cumulative_returns = (1 + portfolio_returns).cumprod()
        sharpe_ratio = (portfolio_returns.mean() - 0.02) / portfolio_returns.std() * np.sqrt(252)  # 연간화된 샤프 비율
        max_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min()

        return {
            'Cumulative Returns': cumulative_returns.iloc[-1] - 1,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown
        }

    def optimize_model(self, performance_metrics):
        """
        모델 성능 평가 결과를 바탕으로 모델 최적화 수행
        """
        # 실제 구현에서는 성능 지표를 바탕으로 모델 파라미터를 조정하는 로직 구현
        if performance_metrics['Sharpe Ratio'] < 1:
            self.ml_model.n_estimators += 50  # 예시: 성능이 낮으면 트리 수 증가

    def run_optimization(self):
        """
        전체 포트폴리오 최적화 프로세스 실행
        """
        self.collect_data()
        self.manage_data_quality()
        self.construct_fama_french_factors()

        self.estimate_factor_betas()
        self.estimate_factor_returns()
        self.calculate_expected_returns()

        self.construct_covariance_matrix()

        self.optimize_portfolio()
        self.calculate_efficient_frontier()
        self.select_optimal_portfolio()

        self.rebalance_portfolio()

        scenario_results = self.run_scenario_analysis()
        robustness = self.evaluate_portfolio_robustness(scenario_results)

        backtest_results = self.backtest_portfolio()

        self.optimize_model(backtest_results)

        return {
            'Optimal Portfolio': dict(zip(self.selected_stocks, self.optimal_portfolio)),
            'Risk Metrics': self.calculate_risk_metrics(self.optimal_portfolio),
            'Portfolio Robustness': robustness,
            'Backtest Results': backtest_results
        }

def main():
    optimizer = PortfolioOptimizer()

    # 투자자 입력 받기
    optimizer.risk_preference = float(input("위험 선호도를 1-10 사이의 숫자로 입력하세요: "))
    optimizer.initial_investment = float(input("초기 투자 금액을 입력하세요: "))
    optimizer.selected_stocks = input("투자하고 싶은 주식 종목을 쉼표로 구분하여 입력하세요: ").split(',')
    initial_weights_input = input("각 종목의 초기 비중을 쉼표로 구분하여 입력하세요 (합이 1이 되어야 함): ").split(',')
    optimizer.initial_weights = np.array([float(x) for x in initial_weights_input])
    optimizer.constraints = input("투자 제약 조건을 입력하세요 (예: 특정 섹터 최대 비중): ")

    # 입력 유효성 검사
    if not np.isclose(np.sum(optimizer.initial_weights), 1.0):
        print("오류: 초기 비중의 합이 1이 되어야 합니다.")
        return

    if len(optimizer.selected_stocks) != len(optimizer.initial_weights):
        print("오류: 선택한 주식 종목 수와 초기 비중 수가 일치하지 않습니다.")
        return

    # 포트폴리오 최적화 프로세스 실행
    results = optimizer.run_optimization()

    # 결과 출력
    print("\n최적화 결과:")
    print(f"최적 포트폴리오:")
    for stock, weight in results['Optimal Portfolio'].items():
        print(f"  {stock}: {weight:.4f}")
    print(f"\n위험 지표:")
    for metric, value in results['Risk Metrics'].items():
        print(f"  {metric}: {value:.4f}")
    print(f"\n포트폴리오 견고성:")
    for metric, value in results['Portfolio Robustness'].items():
        print(f"  {metric}: {value:.4f}")
    print(f"\n백테스트 결과:")
    for metric, value in results['Backtest Results'].items():
        print(f"  {metric}: {value:.4f}")

if __name__ == "__main__":
    main()