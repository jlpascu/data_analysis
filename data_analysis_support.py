from typing import Counter
import pandas as pd
import numpy as np
import copy

np.seterr(divide='ignore')

class DataAnalysis:
    def __init__(self, index_ric, corporate_yeild = None, discount_rate = None, 
                 intermidate_path = "", back_testing = False):
        self.back_testing = back_testing
        self.index_ric = index_ric
        self.path_file = intermidate_path + '/' + index_ric[1:] + '/' 
        self.corporate_yeild = corporate_yeild
        self.discount_rate = discount_rate
        self.constituents_df = self.read_pickle(file_name=index_ric[1:], add_path='pkl/')        
        self.TotalRevenue = self.eliminate_zeros(self.read_pickle(file_name ='TR.TotalRevenue', 
                                                                  add_path ='pkl/'))
        self.OperatingIncome = self.read_pickle(file_name ='TR.OperatingIncome', add_path ='pkl/' )
        self.NetIncomeBeforeTaxes = self.eliminate_zeros(self.read_pickle(file_name= 'TR.NetIncomeBeforeTaxes', add_path='pkl/'))
        self.NetIncomeBeforeExtraItems = self.eliminate_zeros(self.read_pickle(file_name='TR.NetIncomeBeforeExtraItems',
                                                                               add_path='pkl/'))
        self.CashFromOperatingActivities = self.eliminate_zeros(self.read_pickle(file_name='TR.CashFromOperatingActivities',
                                                                                 add_path='pkl/'))
        self.CapitalExpenditures = self.read_pickle(file_name='TR.CapitalExpenditures', add_path='pkl/')
        self.CashFromInvestingActivities = self.eliminate_zeros(self.read_pickle(file_name='TR.CashFromInvestingActivities',
                                                                                 add_path='pkl/'))
        self.TotalReceivablesNet = self.read_pickle(file_name='TR.TotalReceivablesNet', add_path='pkl/')
        self.TotalDebtOutstanding = self.read_pickle(file_name='TR.TotalDebtOutstanding', add_path='pkl/')
        self.TotalAssetsReported = self.read_pickle(file_name='TR.TotalAssetsReported', add_path='pkl/')
        self.TotalEquity = self.read_pickle(file_name='TR.TotalEquity', add_path='pkl/')
        self.TotalLiabilities = self.read_pickle(file_name='TR.TotalLiabilities', add_path='pkl/')
        self.CashAndSTInvestments = self.read_pickle(file_name='TR.CashAndSTInvestments', add_path='pkl/')
        self.TaxRate =  1 - (self.NetIncomeBeforeExtraItems/self.NetIncomeBeforeTaxes)
        self.DilutedEpsExclExtra = self.eliminate_zeros(self.read_pickle(file_name='TR.DilutedEpsExclExtra', add_path='pkl/')) \
                                                                            * (1 - self.TaxRate)
        self.EPSBT_growth = self.get_pct_change(self.DilutedEpsExclExtra)
        self.FCF_growth = self.get_pct_change(self.eliminate_zeros(self.CashFromOperatingActivities + self.CashFromInvestingActivities))
        self.CompanyMarketCap = self.check_backtesting(file_name='TR.CompanyMarketCap')
        self.GICSSubIndustryCode = self.read_pickle(file_name='TR.GICSSubIndustryCode', add_path='pkl/')
        self.PriceClose = self.check_backtesting(file_name='TR.PriceClose')
        self.PE = self.check_backtesting(file_name='TR.PE')
        self.results = pd.DataFrame()  
        self.results_raw = pd.DataFrame()
        self.selected_assets = pd.DataFrame()
        # Create list with analysis that are going to be calculated
        self.item_list = [['Revenues_growth', self.TotalRevenue],                                                           #0
                         ['Earnings_BeforeTaxes&ExtraordinaryItems_growth', self.NetIncomeBeforeTaxes],                     #1
                         ['Operating_Maring', [self.OperatingIncome, self.TotalRevenue]],                                   #2
                         ['Net_Margin_BeforeTaxes&ExtraordinaryItems', [self.NetIncomeBeforeTaxes, self.TotalRevenue]],     #3
                         ['ROIC', [(self.OperatingIncome * (1 - self.TaxRate)), (self.TotalLiabilities \
                                                                                 + self.TotalEquity \
                                                                                 - self.CashAndSTInvestments)]],            #4
                         ['CapExpenditures_over_CFO', [self.CapitalExpenditures, self.CashFromOperatingActivities]],        #5
                         ['Receivables_over_Revenues', [self.TotalReceivablesNet, self.TotalRevenue]],                      #6
                         ['Net_Debt', [self.CashAndSTInvestments, self.TotalDebtOutstanding]],                              #7
                         ['Excluded_Industries', self.GICSSubIndustryCode],                                                 #8
                         ['FCF_over_Revenues', [self.CashFromOperatingActivities + self.CashFromInvestingActivities, 
                                                self.TotalRevenue]],                                                        #9
                         ['Cash_total_assets ', [self.CashAndSTInvestments, self.TotalAssetsReported]]]                     #10
    
    def check_backtesting(self, file_name):
        '''
        Run specific parts of the analysis when we are not running a backtesting
        
        Args:
            file_name: name of the pickle file to be openned
        Retruns:
            df: DataFrame with the financial data
        '''
        if self.back_testing == False:
            df = self.read_pickle(file_name= file_name, add_path='pkl/')
            return df
    
    def eliminate_zeros(self, df):
        '''
        Substitutes 0 by numpy NaN
        
        Args:
            df: DataFrame with data to be analyzed
        Returns:
            df: DtaFrame with no zeros
        '''
        df = df.where(df !=0, np.nan)
        return df

    def get_pct_change(self, df):
        '''
        Calculates average growth over DF columns

        Args:
            df: DF containg financial data
        Returns:
            mean: PD series containing average result
        '''
        pct_change = df.pct_change(axis = 'columns')
        mean = pct_change.mean(axis = 1) * 100
        return mean                                                   

    def read_pickle(self, file_name, add_path = ''):
        '''
        Reads pickle file and returns adata

        Args:
            file_name : name of the file to be opened
            add_path : intermidate path that we should add to read pickle file
        Returns:
            data_df: DF or Series containing financial data
        '''
        path_file = self.path_file + 'data_downloaded/final_data/' + add_path + file_name + '.pkl'
        data_df = pd.read_pickle(path_file)
        # In some cases, when we are running a backtesting, RIC codes are 
        # duplicated. This is probably because one of the companies does not
        # longer exists. As numbers are equal for both names, we will eliminate
        # one of the copies. For index file, RIC are not in the DF index.
        if file_name == self.index_ric[1:]:
            data_df = data_df.loc[~data_df['Constituent RIC'].duplicated(), :]
        data_df = data_df.loc[~data_df.index.duplicated(), :]
        return data_df

    def get_last_column(self, df):
        '''
        Selects las column of DF

        Args:
            df: DF containing financial data
        Retruns:
            last_column: Series containg las DF column
        '''
        # Select last column
        last_column = pd.DataFrame(df.iloc[:,-1])
        # Change Series column
        last_column.columns = [None]
        return last_column

    def get_rate_return(self):
        '''
        Calculates Warren buffet return on "equity bonds" which is EPSBT / Last Price

        Args:
            None
        Returns:
            rate_return: Pandas Series containing pretaxe reate of return for every asset. 
        '''
        # get last epsbt
        last_eps = self.get_last_column(df = self.DilutedEpsExclExtra)
        # get last price
        last_price = self.PriceClose
        # Set column to None
        last_price.columns = [None]
        # Divide Series
        rate_return = (last_eps / last_price) * 100
        return rate_return

    def get_intrinsic_value(self):
        '''
        Calculates intrinsic value over and investment period of 20 years. 
        EPSBT grow at Warrent Buffett equity bonds and are dicounted at corporate yeild.
        '''
        # get last epsbt
        last_eps = self.get_last_column(df=self.DilutedEpsExclExtra)
        # get web equity bond yeild
        rate_return = self.get_rate_return() / 100
        # calculate future epsbt
        future_eps = last_eps * ((1+rate_return) ** self.investment_period)
        # discount epsbt to pv
        discounted_eps = future_eps * ((1+self.corporate_yeild)**(self.investment_period * -1))
        # Get last PE
        pe = self.get_last_column(df = self.PE)
        # Multiply discounted EPSBT by Price to Earings
        value_share = discounted_eps * pe
        # Returuen intrinsic value
        return value_share

    def smaller_better(self, margin):
        '''
        Calculates ratio inverse.

        Args:
            margin: DF containing ratios
        Returns:
            inverse_margin: DF containing inverse ratios
        '''
        margin = margin / 100
        inverse_margin = (1 - margin) * 100
        return inverse_margin  

    def get_score(self):
        '''
        Calculates final score for every asset class. 
        Score is calculated base on results DataFrame, where scaled data
        has been saved. 
        Args:
            None
        Retruns:
            None
        '''
        score = self.results.sum(axis = 1)
        self.results['score'] = score
        self.results_raw['score'] = score

    def add_to_results(self, df, item, item_num, raw = False):
        '''
        Add analysis to a final results DataFrame
        Args:
            df: DataFrame with the analysis
            item_num: number of analysis that have been done
            raw: determinse specific results DataFrame. Raw = True saves
                non-scaled data
        Returns:
            None
        '''
        # Check if it is non-sacled data
        if raw == False:
            # For the first analysis we must create our results DataFrame
            if item_num == 0:
                # Create DataFrame
                self.results = pd.DataFrame(df.iloc[:,-2])
                # Change column name
                self.results = self.results.rename(columns = {'average': item})
            else:
                # Save results to results DF
                self.results[item] = df.iloc[:,-2]
        # Check if it is non-sacled data
        elif raw == True:
            # For the first analysis we must create our results DataFrame
            if item_num == 0:
                # Create DataFrame
                self.results_raw = pd.DataFrame(df.iloc[:,-2])
                # Change column name
                self.results_raw = self.results_raw.rename(columns = {'average': item})
            else:
                # Save results to results DF
                self.results_raw[item] = df.iloc[:,-2]
    
    def save_results(self, df, file_name, raw = False):
        '''
        Saves DataFrame to different formats

        Args:
            df: DataFrame to be saved
            file_name: name given to the saved file
            raw: determinse specific folder. Raw = True saves
                non-scaled data
        Returns:
            None
        '''
        if raw == True:
            file_path = self.path_file + 'results/'
            df.to_pickle(file_path + 'pkl/' + 'raw/' + file_name + '.pkl')
            df.to_csv(file_path + 'csv/' + 'raw/' + file_name + '.csv')
            df.to_excel(file_path + 'xlsx/' + 'raw/' + file_name + '.xlsx')
        else:
            file_path = self.path_file + 'results/'
            df.to_pickle(file_path + 'pkl/' + 'scale/' + file_name + '.pkl')
            df.to_csv(file_path + 'csv/' + 'scale/' + file_name + '.csv')
            df.to_excel(file_path + 'xlsx/' + 'scale/' + file_name + '.xlsx')

    def sort_values(self, df, column_name, asc = False):
        '''
        Sorts DataFrame values by a specified column.

        Args:
            column_name: column name by which DataFrame values will be sorted
            asc: ascending or descending order when sorting values by average
        Returns:
            df: Original DataFrame sorted by the specified column.
        '''
        df = df.sort_values(by=column_name, ascending = asc)
        return df

    def add_company_names(self, df, filter_assets = False):
        '''
        Adds companies' full name to df with financial data. 
        Full name complementes Eikon RIC when identifying companies. 

        Args:
            df: DataFrame with assets financial information
            filter_assets: bolean variable. Takes into consideration the assets that
            have been filter when adding names to final results DFs
        Returns:
            df: Original DataFrame with an additional column where companies' 
            names are stored.
        '''
        # Check if this is a pandas Series
        if type(df) == pd.core.series.Series:
            # Convert Series to DataFrame
            df = pd.DataFrame(df, columns=['average'])
        # Sort by index (RIC codes)
        df = df.sort_index(ascending = True)
        # Order index_ric DataFrame by RIC codes
        index_df = self.constituents_df.sort_values(by = 'Constituent RIC', ascending = True)
        # In case df has been filter, we need to filter constituents index too
        if filter_assets == True:
            # Set RIC as index
            index_df = index_df.set_index('Constituent RIC', drop = False)
            # Filter names
            index_df = index_df.loc[self.selected_assets,:]
            # Add comany names to our DataFrame
            df['company_names'] = index_df.loc[:,'Constituent Name']
        elif filter_assets == False:
            # Add comany names to our DataFrame
            df['company_names'] = index_df.loc[:,'Constituent Name'].values
        return df

    def count_constant_increases(self, df):
        '''
        Counts postive increses over year

        Args:
            df: DF with financial data
        Returns:
            count_df: DF with analysis results
        '''
        # read pd rows and columns lenght
        total_assets = df.shape[0]
        total_years = df.shape[1]
        # Fill NA with 0 to aovid moolean exceptions
        df = df.fillna(value = 0)
        # run for all assets
        for asset in range(0, total_assets, 1):
            count = 0
            # run for every fiscal year
            for year in range(0, (total_years - 1), 1):
                # compare year + 1 against year - 1
                if df.iloc[asset,(year + 1)] > df.iloc[asset,year]:
                    count = count + 1
            # save on pandas series
            if asset == 0:
                # Create pandas Series with first asset
                count_df = pd.Series(count, index = [df.index[asset]])
            else:
                # Append rest of the assets
                count_df = count_df.append(pd.Series(count, index = [df.index[asset]]))
        return count_df

    def constant_increase_analysis(self, df, item, count):
        '''
        Starts constant increase analysis. 

        Args:
            df: DF containg financial data
            item: name of the analysis
            count: position of current analysis over all analysis coded
        Returns:
            None
        '''
        # Count constant increase
        count_df = self.count_constant_increases(df)
        # Continue with standard analysis
        self.base_analysis(count_df, item, count)

    def scale(self, data, negative, count = None, column_name = None):
        '''
        Scale values over 100

        Args:
            data: pandas DataFrame or Series with the financial data to 
            be scaled
            column_name: In case data is a DataFrame, we must indicate which
            column will be used to scale values
            negative: indicates if inverse has to be calculated
        Returns:
            result: Pandas Series with scaled values
            data: pandas DataFrame with values scaled by the column specified. 
        '''
        # Check if this is a pandas Series. 
        if type(data) == pd.core.series.Series:
            # Calculate sum of all averages
            total_sum = abs(data.sum())
            # Divide every average by the total sum and
            # multiply by 100
            result = (data / total_sum) * 100
            return result
        else:
            if negative == True:
                data[column_name] = self.smaller_better(data[column_name])
            # Calculate sum of all averages
            total_sum = abs(data.loc[:,column_name].sum(axis=0))
            # Divide every average by the total sum and
            # multiply by 100
            if count == 1:
                # For some ratios, we will like to have higher importance. We a applyy to them
                # a mutliplier
                multiplier = 1
                print('We apply to multipler of', multiplier)
                data[column_name] = ((data.loc[:,column_name] / total_sum) * 100) * multiplier
            else:
                data[column_name] = (data.loc[:,column_name] / total_sum) * 100
        return data
        
    def get_average(self, df):
        '''
        Calculates average per row on a DF

        Args:
            df: DataFrame used for calculations
        Returns:
            df: DataFrame with average results added as an additional column
        '''
        # Calculate average and adds results as an additional column
        df['average'] = df.mean(axis=1)
        # When the ratio being calculated has to be the smaller
        # the better, the inverse is calculated
        return df

    def get_valuation_analysis(self):
        '''
        Adss several valuation analysis to results and results_raw DF

        Args:
            None
        Returns:
            None
        '''
        # Include last price
        self.results['Last_Price'] = self.PriceClose
        self.results_raw['Last_Price'] = self.PriceClose
        # Get last_epsbt
        self.results['last_epsbt'] = self.get_last_column(df=self.DilutedEpsExclExtra)
        self.results_raw['last_epsbt'] = self.get_last_column(df=self.DilutedEpsExclExtra)
        # Add EPSBT growth
        self.results['EPSBT_growth'] = self.EPSBT_growth 
        self.results_raw['EPSBT_growth'] = self.EPSBT_growth 
        # Add capitalized intrinsic value
        self.results['EPSBT_capitalized'] = (self.results['last_epsbt'] * (1 + self.EPSBT_growth/100)) / self.corporate_yeild
        self.results_raw['EPSBT_capitalized'] = (self.results_raw['last_epsbt'] * (1 + self.EPSBT_growth/100)) / self.corporate_yeild
        # Compare capitalized value vs last price
        self.results['buy_EPSBT_capitalized'] = self.results['EPSBT_capitalized'] > self.results['Last_Price']
        self.results_raw['buy_EPSBT_capitalized'] = self.results_raw['EPSBT_capitalized'] > self.results_raw['Last_Price']
        # Get last_FCF
        self.results['last_FCF'] = self.get_last_column(df=self.CashFromOperatingActivities + self.CashFromInvestingActivities)
        self.results_raw['last_FCF'] = self.get_last_column(df=self.CashFromOperatingActivities + self.CashFromInvestingActivities)
        # Add FCF growth
        self.results['FCF_growth'] = self.FCF_growth 
        self.results_raw['FCF_growth'] = self.FCF_growth 
        # FCF valuation
        self.results['FCF_valuation'] = (self.results['last_FCF'] * (1 + self.FCF_growth/100)) / self.discount_rate
        self.results_raw['FCF_valuation'] = (self.results_raw['last_FCF'] * (1 + self.FCF_growth/100)) / self.discount_rate
        # Company marketcap
        self.results['CompanyMarketCap'] = self.CompanyMarketCap
        self.results_raw['CompanyMarketCap'] = self.CompanyMarketCap
        # Compare capitalized value vs last price
        self.results['buy_FCF_capitalized'] = self.results['FCF_valuation'] > self.results['CompanyMarketCap']
        self.results_raw['buy_FCF_capitalized'] = self.results_raw['FCF_valuation'] > self.results_raw['CompanyMarketCap']
        # Add last PE
        self.results['PE'] = self.get_last_column(df = self.PE)
        self.results_raw['PE'] = self.get_last_column(df = self.PE)
    
    def get_margin(self, num, denom):
        '''
        Calculates ratio num / denom

        Args:
            num: numerator
            denom: denominator
        Returns:
            margin :  (num/denom)*100
        '''
        # Sort both DF 
        num = num.sort_index()
        denom = denom.sort_index() 
        # Substitue negative values of denom  to NaN. 
        # In those cases our ratio does not apply
        denom = denom.where(denom > 0, other = np.nan)
        # Divide both DF
        margin = num / denom
        # multiply results by 100
        margin = margin * 100
        return margin      
    
    def growth_analysis(self, df, item, count, negative = False):
        '''
        Calculates log returns between columns/years
        Assets with negative values in any year are excluded form computation

        Args:
            df: fiancial data
            item: item being calculated
            count: number of analysis performed
            negative: indicates if inverse has to be calulated
        Returns:
            None
        '''
        # Transpose df
        df_trans = df.transpose()
        # Check lenght
        df_lenght = len(df_trans)
        # Create boolean mask for asset with positve data on every year
        boolean_mask_postive = (df_trans > 0).sum() == df_lenght
        # Create boolean mask for the rest of assets
        boolean_mask_negative = (df_trans > 0).sum() != df_lenght
        # Select assets according to boolean_mask_postive
        df_filterd_positive = df_trans.loc[:, boolean_mask_postive]
        # Calculate log returns and transpose df
        log_returns = (np.log(df_filterd_positive.astype(float)).diff() * 100).transpose()
        # Create df with nan for assets that had been excluded
        df_filterd_negative = df_trans.loc[:, boolean_mask_negative].transpose()
        nan_df = pd.DataFrame(np.nan, index=df_filterd_negative.index, columns=log_returns.columns)
        # Append empty rows to log returns dataframe
        log_returns = log_returns.append(nan_df)
        # Calculate average
        log_returns = self.get_average(log_returns)
        # Get standard analysis
        self.base_analysis(log_returns, item, count, negative)
    
    def include_margin(self, df, item, count, negative = False):
        '''
        Calculates ratio numerator over denominator, and average
        Args:
            df: list including [numerator, denominator]. Make sure order is correct
            item: item being calculated
            count: number of analysis performed
            negative: indicates if inverse has to be calulated
        Returns:
            None
        '''
        # Extract tupple
        (num, denom) = df
        # Calculate margin
        df = self.get_margin(num=num, denom=denom)
        # Calculate average
        df = self.get_average(df)
        # Get standard analysis
        self.base_analysis(df, item, count, negative)
    
    def net_debt_filter(self, df, item):
        '''
        Calculates net debt and filters results and results_raw
        DF by assets that do not have debt. 

        Args:
            df: DataFrame with financial data
            item: name of the ratio being calculated
        Returns:
            None
        '''
        # Extract tupple
        (cash_df, debt_df) = df
        # Sort both DF 
        cash_df = cash_df.sort_index()
        debt_df = debt_df.sort_index()
        # Calculate net debt
        net_debt_df = cash_df - debt_df
        # Calculate average
        df = self.get_average(net_debt_df)
        # Add company names to RIC codes
        df = self.add_company_names(df)
        # Sort values by specified column name
        df = self.sort_values(df, column_name='average')
        # Save DataFrame to diffrerent formats
        self.save_results(df, file_name=item + '_raw', raw=True)
        # Create boolean mask
        boloean_mask = df['average'] > 0
        # Select assets withou debt
        self.selected_assets = df.loc[boloean_mask, 'average'].index
        # Filter assets on results df
        self.results = self.results.loc[self.selected_assets,:]
        self.results_raw = self.results_raw.loc[self.selected_assets,:]
    
    def industry_filter(self, df):
        '''
        Excludes certain assets from our analysis because they belong to an
        industry that does not perfrom well in the long term. 

        Args:
            df: DataFrame with financial data
        Returns:
            None
        '''
        # Index must be converted to list because index has no remouve fuction
        self.selected_assets = self.selected_assets.tolist()
        # Create indsutry code list. Codes according to GICS Subsindustry classification
        # 35201010 --> Biotechnology
        # 60101080 --> Specialized REITs
        industry_code_list = [35201010, 60101080, 60101050, 60101070, 60101060, 40201030, 40203010]
        # Run for every industry code
        for industry_code in industry_code_list:
            # Create boolean mask
            boolean_mask = df.iloc[:,0] == industry_code
            # Select assets that have been exluded because of the sectors they belong to
            exluded_sectors = self.GICSSubIndustryCode.loc[boolean_mask,:].index
            # Ruen for al the excuded assets because of their industry
            for asset in exluded_sectors:
                # Check if the assets have already been exlcuded (previous filters)
                if asset in self.selected_assets:
                    # Remove in case the asset is still in the selected assets list
                    self.selected_assets.remove(asset)
        # Filter assets on results df
        self.results = self.results.loc[self.selected_assets,:]
        self.results_raw = self.results_raw.loc[self.selected_assets,:]
    
    def base_analysis(self, df, item, count, negative=False):
        '''
        Runs serveral adjustments that are common to all financial items
        (Scale results, add company names, sort values, save results and 
        add results to a final DF)

        Args:
            df: DataFrame with financial data
            item: name of the ratio being calculated
            negative: indicates if inverse has to be calculated
        Returns:
            None
        '''
        # We have interest in saving the non-scale financial data. 
        # We create a deep copy of our original data to include it 
        # for analysis porpuses. 
        unscale_df = copy.deepcopy(df)
        # Scale data
        scale_df = self.scale(data = df, negative=negative, count =  count, column_name='average')
        # For some ratios, scale values can be extreme and distortionate total score. We
        # set a cap and floor for scale values on every ratio. We will not scale counting 
        # pandas Series
        if type(scale_df) != pd.core.series.Series:
            mean = scale_df['average'].mean()
            std = scale_df['average'].std()
            upper_band = mean + 3 * std
            lower_band = mean - 3 * std
            # Absolut limit measure
            if upper_band > 5:
                upper_band = 5
            if lower_band < -5:
                lower_band = -5
            scale_df['average'] = scale_df.loc[:,'average'].clip(lower = lower_band, upper = upper_band)
        # Add company names to RIC codes
        unscale_df = self.add_company_names(unscale_df)
        scale_df = self.add_company_names(scale_df)
        # Sort values by specified column name
        unscale_df = self.sort_values(unscale_df, column_name='average')
        scale_df = self.sort_values(scale_df, column_name='average')
        # Save DataFrame to diffrerent formats
        self.save_results(unscale_df, file_name=item + '_raw', raw=True)
        self.save_results(scale_df, file_name = item)
        # Add analysis to results DataFrame
        self.add_to_results(unscale_df, item, item_num=count, raw = True)
        self.add_to_results(scale_df, item, item_num = count, raw = False)
        
    def run_analysis(self):
        '''
        Performs the analysis, according to the position that each ratio 
        has in self.item_list.
    
        Args:
            None
        Returns:
            None
        '''
        # Run analysis for each reatio included in the self.item_list
        for count, (item, df) in enumerate(self.item_list):
            if (count == 0) or (count == 1):
                # This analysis is performed for ratios in which the higher
                # the result, the more positive is for the company
                print('We run growth positive analysis for', item)
                self.growth_analysis(df, item, count, negative=False)
            elif (count == 6):
                # This analysis is performed for ratios in which the lower
                # the result, the more positive is for the company
                print('We run margin negative analysis for', item)
                self.include_margin(df, item, count, negative =True)
            elif (count == 2) or (count == 3) or \
                 (count == 4) or (count == 9) or \
                 (count == 10) or (count == 5):
                # This analysis is performed for ratios in which the higher
                # the result, the more positive is for the company
                print('We run margin positive analysis for', item)
                self.include_margin(df, item, count, negative=False)
            elif count == 7:
                print('We run net debt filter for', item)
                # Filter results by net debt
                self.net_debt_filter(df, item)
            elif count == 8:
                print('We exclude certain industries for', item)
                self.industry_filter(df)
        # Calculate final score
        self.get_score()
        # Add valuation analysis. Run only when we are not running a backtesting
        if self.back_testing == False:
            self.get_valuation_analysis()
        # Add names to final result DataFrame
        self.results = self.add_company_names(self.results, filter_assets =  True)
        self.results_raw = self.add_company_names(self.results_raw, filter_assets = True)
        # Save final analysis to different formats under name 'results'
        self.save_results(df = self.results_raw, file_name='results_raw', raw=True)
        self.save_results(df = self.results, file_name = 'results')
        # Save some intermidate calculations
        self.save_results(self.TaxRate, file_name='TaxRate' + '_raw', raw=True)
        self.save_results(self.TaxRate, file_name = 'TaxRate')
        self.save_results(self.DilutedEpsExclExtra, file_name='EPSBT' + '_raw', raw=True)
        self.save_results(self.DilutedEpsExclExtra, file_name = 'EPSBT')
        self.save_results((self.CashFromOperatingActivities + self.CashFromInvestingActivities), 
                          file_name='FCF' + '_raw', raw=True)
        self.save_results((self.CashFromOperatingActivities + self.CashFromInvestingActivities), file_name = 'FCF')
