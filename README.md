# data_analysis
On this depository you can find a long-term investment algorithm strategy that assigns a score on every asset that belongs to an equity index based on the specify financial ratios.

Run file ‘run_file.py’ to run the code.  

data_analysis_support.py is the module used. Make sure it is installed in your venv.  

This code will assign a score to all the assets that belong to an equity index based on the financial ratios passed when creating the DataAnalysis classe. See data_analysis_support.py file to change the financial ratios.  

Additionally, a simple valuation is calculated for every asset based on the following formulas: 

Company value = (FCF * (1+ FCF growht)) / discount_rate 

Share_value = (last_EPS_before_taxes * (1 + last_EPS_before_taxes)) / corporate_yeild 

Discount_rate and corporate_yeild can be updated on the run file. This will be your personal opportunity cost 

Make sure you have downloaded the data need with the codes that you can find on download_data_from_eikon repository. 


