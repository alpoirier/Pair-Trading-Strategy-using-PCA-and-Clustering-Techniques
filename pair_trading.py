import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import OPTICS
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts
from itertools import combinations, chain
import warnings


class PCA_Clustering_Pair_Strategy:

    
    """
    
    Initialise
    
    """

    def __init__(self, data, n_components_):
     

        self.prices = data
        self.securities = self.prices.columns
        self.returns = self.prices.pct_change()[1:]
        
        ####### PCA PART ###########
        
        Scaler=StandardScaler()
        random_state: int = 42
        
        # PCA pipeline
        pipe = Pipeline([
            # Normalize raw data via user input scaler
            ('scaler', Scaler),
            # Perform PCA on scaled returns
            ('pca', PCA(n_components=n_components_, random_state=random_state))
            ])
        
        # Reduced transform of returns from PCA
        self.returns_reduced = pipe.fit_transform(self.returns) 
        
        # Components generated from PCA
        self.components_ = pipe['pca'].components_
        
        # Number of components of PCA
        self.n_components_ = pipe['pca'].n_components_
        
        # Variance explained by PCA
        self.explained_variance_ratio_ = pipe['pca'].explained_variance_ratio_        
        
        
        
        ####### Clustering of PCs PART ###########
        
        # Initialize and fit OPTICS cluster to PCA components
        clustering = OPTICS()
        clustering.fit(self.components_.T)
        
        # Create cluster data frame and identify trading pairs
        clusters = pd.DataFrame({'security': self.securities,
                                 'cluster': clustering.labels_})
        
        clusters = clusters[clusters['cluster'] != -1]

        # Group securities by cluster and flatten list of combination lists
        groups = clusters.groupby('cluster')
        combos = list(groups['security'].apply(combinations, 2))  # All pairs
        pairs = list(chain.from_iterable(combos))  # Flatten list of lists

        print(f"Found {len(pairs)} potential pairs")

        
        # Potential pairs found from OPTICS clusters
        self.pairs = pd.Series(pairs)
        
        self.cluster_labels = clustering.labels_
        
        


    
    
    
    
    
    """
    
    Testing each pair for cointegration
    
    """
    
    
    def calc_eg_norm_spreads(self):
       

        engle_granger_tests = []
        norm_spreads = []

        
        # Test each pair for cointegration
        for pair in self.pairs:

            security_0 = self.prices[pair[0]]
            security_1 = self.prices[pair[1]]

            
            # Get independent and dependent variables for OLS calculation 
            
            pvalue, x, y = PCA_Clustering_Pair_Strategy.get_ols_variables(security_0, security_1)
            
            # Also retrieve corresponding pvalue for Engle-Granger tests 
            
            engle_granger_tests.append(pvalue)
            
            

            # Get parameters and calculate spread
            model = sm.OLS(y, x)
            result = model.fit()
            alpha, beta = result.params[0], result.params[1]

            spread = y - (alpha + beta*x.T[1])
            
            
            # Z score function used to normalize the spread
            
            norm_spread = PCA_Clustering_Pair_Strategy.calc_zscore(spread)
            norm_spreads.append(norm_spread)

        # Convert spreads from list to dataframe
        norm_spreads = pd.DataFrame(np.transpose(norm_spreads),
                                    index=self.prices.index)

        self.norm_spreads = norm_spreads
        self.engle_granger_tests = pd.Series(engle_granger_tests)
    


    # Engle Granger P values + OLS variables  
    
    @staticmethod
    def get_ols_variables(security_0,
                          security_1):
      

        test_0 = ts.coint(security_0, security_1)
        test_1 = ts.coint(security_1, security_0)

        t_stat_0, pvalue_0 = test_0[0], test_0[1]
        t_stat_1, pvalue_1 = test_1[0], test_1[1]
  

        if abs(t_stat_0) < abs(t_stat_1):
            pvalue = pvalue_0
            x = sm.add_constant(np.asarray(security_1))
            y = np.asarray(security_0)
        else:
            pvalue = pvalue_1
            x = sm.add_constant(np.asarray(security_0))
            y = np.asarray(security_1)

        return pvalue, x, y
    
    
    @staticmethod
    def calc_zscore(spread):
        zscore = (spread - np.mean(spread))/np.std(spread)
        return zscore   
  

    
    """
    
    Mean Reversion using Hurst Exponent
    
    """
    
    

    def calc_hurst_exponents(self):
       

        hurst_exponents = []

        # Calculate Hurst exponents and generate series
        for col in self.norm_spreads.columns:
            hurst_exp = PCA_Clustering_Pair_Strategy.hurst(self.norm_spreads[col].values)
            hurst_exponents.append(hurst_exp)

        self.hurst_exponents = pd.Series(hurst_exponents)
        

    @staticmethod
    def hurst(norm_spread):

        # Create the range of lag values
        lags = range(2, 100)

        # Calculate the array of the variances of the lagged differences
        diffs = [np.subtract(norm_spread[l:], norm_spread[:-l]) for l in lags]
        tau = [np.sqrt(np.std(diff)) for diff in diffs]
        
        # Use a linear fit to estimate the Hurst Exponent
        poly = np.polyfit(np.log(lags), np.log(tau), 1)

        # Return the Hurst exponent from the polyfit output
        H = poly[0]*2.0

        return H
    
  


    """
    
    Time to Mean Revert with Half Life
    
    """        
        
        
    def calc_half_lives(self):
    

        self.half_lives = self.norm_spreads.apply(PCA_Clustering_Pair_Strategy.half_life)
        

    @staticmethod
    def half_life(norm_spread):
     
        lag = norm_spread.shift(1)
        lag[0] = lag[1]

        ret = norm_spread - lag
        lag = sm.add_constant(lag)

        model = sm.OLS(ret, lag)
        result = model.fit()
        half_life = -np.log(2)/result.params.iloc[1]

        return half_life
    
        
        
    """
    
    Testing how often the pair crosses
    
    """  
    
        
    def calc_avg_cross_count(self, trading_year = 365):
     

        # Find number of years
        n_days = len(self.prices)
        n_years = n_days/trading_year

        # Find annual average cross count
        cross_count = self.norm_spreads.apply(PCA_Clustering_Pair_Strategy.count_crosses)
        self.avg_cross_count = cross_count/n_years




    @staticmethod
    def count_crosses(norm_spread, mean = 0):
     

        curr_period = norm_spread
        next_period = norm_spread.shift(-1)
        count = (
            ((curr_period >= mean) & (next_period < mean)) |  # Over to under
            ((curr_period < mean) & (next_period >= mean)) |  # Under to over
            (curr_period == mean)
            ).sum()

        return count

    
    """
    
    Filtering 
    
    """    
    
    
    def filter_pairs(self,
                     max_pvalue = 0.05,
                     max_hurst_exp = 0.5,
                     max_half_life = 252.0,
                     min_half_life = 1.0,
                     min_avg_cross = 12.0):




        # Generate summary dataframe of potential trading pairs
        pairs_df = pd.concat([self.pairs,
                              self.engle_granger_tests,
                              self.hurst_exponents,
                              self.half_lives,
                              self.avg_cross_count],
                             axis=1)
        
        pairs_df.columns = ['pair',
                            'pvalue',
                            'hurst_exp',
                            'half_life',
                            'avg_cross_count']

        
        # Find pairs that meet user defined criteria
        filtered_pairs = pairs_df.loc[
            # Significant Engle-Grange test AND
            (pairs_df['pvalue'] <= max_pvalue) &
            # Mean reverting according to Hurst exponent AND
            (pairs_df['hurst_exp'] < max_hurst_exp) &
            # Half-life above minimum value AND
            # Half-life below maximum value AND
            ((pairs_df['half_life'] >= min_half_life) &
             (pairs_df['half_life'] <= max_half_life)) &
            # Produces sufficient number of trading opportunities
            (pairs_df['avg_cross_count'] >= min_avg_cross)]

        self.pairs_df = pairs_df
        self.filtered_pairs = filtered_pairs

        if len(self.filtered_pairs) == 0:
            print("No tradable pairs found. Try relaxing criteria.")
        else:
            n_pairs = len(self.filtered_pairs)
            print(f"Found {n_pairs} tradable pairs!")
    
    
    """
    
    PLOTS
    
    """
    
    def plot_pair_price_spread(self, idx):
   
    

        fontsize = 20
        securities = self.pairs[idx]

        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(20, 10))

        
        # first security (left axis)
        security = securities[0]
        color = 'tab:blue'
        axs[0].plot(self.prices[security], color=color)
        axs[0].set_ylabel(security, color=color, fontsize=fontsize)
        axs[0].tick_params(axis='y', labelcolor=color)
        axs[0].set_title('pair_'+str(idx)+' prices', fontsize=fontsize)

        # second security (right axis)
        security = securities[1]
        color = 'tab:orange'
        axs2 = axs[0].twinx()
        axs2.plot(self.prices[security], color=color)
        axs2.set_ylabel(security, color=color, fontsize=fontsize)
        axs2.tick_params(axis='y', labelcolor=color)

        # plot spread
        axs[1].plot(self.norm_spreads[idx], color='black')
        axs[1].set_ylabel('spread_z_score', fontsize=fontsize)
        axs[1].set_xlabel('date', fontsize=fontsize)
        axs[1].set_title('pair_'+str(idx)+' normalized spread',
                         fontsize=fontsize)
        axs[1].axhline(0, color='blue', ls='--')
        axs[1].axhline(1, color='r', ls='--')
        axs[1].axhline(-1, color='r', ls='--')

        fig.tight_layout()
    
    
    """
    
    All in method
    
    """        
    
    
    def perform_selection(self):
        
        self.calc_eg_norm_spreads()
        self.calc_hurst_exponents()
        self.calc_half_lives()
        self.calc_avg_cross_count()
        self.filter_pairs()
        print(self.filtered_pairs)