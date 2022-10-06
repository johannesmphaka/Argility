from cgi import print_exception
import streamlit as st
import pandas as pd
# from streamlit_option_menu import option_menu

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings('ignore')
from bokeh.plotting import figure, show, output_file, output_notebook, save
from bokeh.palettes import Spectral11, colorblind, Inferno, BuGn, brewer
from bokeh.models import HoverTool, value, LabelSet, Legend, ColumnDataSource,LinearColorMapper,BasicTicker, PrintfTickFormatter, ColorBar
import datetime as dt
import streamlit.components.v1 as components
from bokeh.models import BoxAnnotation, Span
from bokeh.models import Slope
import joblib


import mlflow
        
mlflow.set_tracking_uri("http://127.0.0.1:5000") 

st.set_page_config(layout="wide")






# st.title('ECS Baseline')
# st.header('Exploratory Data Analysis')


# Using "with" notation
with st.sidebar:
    
    products = pd.read_csv('Senior Data Assignment.csv')
    products = ['Product_ID: ' + '' + str(i) for i in products['sku_cde'].unique()]
    
    
    products1 = ["11990782", ]
    
    option = st.selectbox(
    'Select Product ID',
    products)
    
    
    

if option == 'Product_ID: 62875832':
    st.title('Product_ID: 62875832')
   
    loaded_model_prom  = joblib.load('Modelss/model_regular_62875832.pkl') 
    loaded_model = joblib.load('Modelss/model_promo_62875832.pkl') 
    
    
    # loaded_model = mlflow.pyfunc.load_model(logged_model)
    
    
    
    
    Price_Elasticity_regular = np.round(loaded_model.coef_[0], 2)
    Price_Elasticity_promo = np.round(loaded_model_prom.coef_[0], 2)
    
    st.subheader(f'Price Elasticity for regular price: {Price_Elasticity_regular}')
    st.subheader(f'Price Elasticity for promo price: {Price_Elasticity_promo}')
    
    
    df = pd.read_csv('Senior Data Assignment.csv')
    
    columns = [i for i in df.columns]
    sku_cde = [i for i in df['sku_cde'].unique()]
    df_regular = df[columns[0:5]]
    df_regular = df_regular.dropna()
    
    
    
    # product_ = pd.read_csv('Senior Data Assignment.csv')
    product_ = df_regular[df_regular['sku_cde']==62875832]
    product_['date_week'] = pd.to_datetime(product_['date_week'])
    # product_['year'] = pd.to_datetime(product_['date_week']).dt.year
    
    # st.dataframe(product_)
    ##pr---- price
    
    maxx = [i for i in product_[['regular_price']].max()][0]
    minn = [i for i in product_[['regular_price']].min()][0]
    
    ### avarage cost price for the product
    mean_price = np.round(product_['cost_price'].mean(), 2)

    
    mid = (maxx + minn)/2
    
    with st.sidebar:
        
        pr = st.sidebar.slider('Choose price', minn, maxx+1, mid)
        st.write('For Product ID: 111708109, we see that max price was 186 when the optimal price could be 126.47 for maximum profit . this means the Product was loosing in profit. but if they had sold for 126 the sales may decrease, quantity been sold will decrease off course but profit will increase.')
    
        st.write('**Please note that all the products can be interpreted in an similar manner**')
    
        st.write('Now that we have these new price points for each items, once they are set up is important to continuously monitoring the sales and profits made out of these items.')


        st.write('**Elasticities interpretation**')
        
        st.write('A 10% price decrease in Product ID: 111708109 , it increases sales demand by 172.8% or a 10% price increase in Product ID: 62875832, it decreases sales demand by 172.8%')


    
    col1, col2, col3 = st.columns(3)
    # col1.metric("Temperature", "70 °F", "1.2 °F")
    # col2.metric("Wind", "9 mph", "-8%")
    # col3.metric("Humidity", "86%", "4%")
    
    #############################################################################################
    with col1:
        
        
        
      #####$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        # loaded_model = mlflow.pyfunc.load_model(logged_model)
        
        current_price = pd.DataFrame(columns= [ "PRICE", "QUANTITY"])
        current_price['PRICE'] = [pr]
        current_price['QUANTITY'] = np.round(loaded_model.predict(pd.DataFrame([pr])), 2)
        current_price['PROFIT'] = np.round((current_price["PRICE"] - mean_price) * current_price['QUANTITY'], 2)
        
        
        test = pd.DataFrame(columns= [ "PRICE", "QUANTITY"])
        
        # price in step of 0.01 and denominations increase in that sense
        test["PRICE"] = np.arange(minn, maxx+1, 0.01)
        test["QUANTITY"] = loaded_model.predict(pd.DataFrame(test['PRICE']))
        
        # Profit
        test['PROFIT'] = (test["PRICE"] - mean_price) * test['QUANTITY'] 
        
        ind = np.where(test['PROFIT'] == test['PROFIT'].max())[0][0]
        indata = test.loc[[ind]]
        
        
        pricee = np.round([i for i in indata['PRICE']][0],2)
        profitt = np.round([i for i in indata['PROFIT']][0], 2)
        quantity = np.round([i for i in indata['QUANTITY']][0], 2)
       
#########################################################################################



        test_best = pd.DataFrame(columns= [ "PRICE", "QUANTITY"])
        test_best["PRICE"] = np.arange(minn-20, maxx, 0.01)
        test_best["QUANTITY"] = loaded_model.predict(pd.DataFrame(test_best['PRICE']))
        
        # Profit
        # 
        test_best['PROFIT'] = (test_best["PRICE"] - mean_price) * test_best['QUANTITY'] 
        
        
        ind_best = np.where(test_best['PROFIT'] == test_best['PROFIT'].max())[0][0]
        indata_best = test_best.loc[[ind_best]]
        
        
        price_best = np.round([i for i in indata_best['PRICE']][0],2)
        profit_best = np.round([i for i in indata_best['PROFIT']][0],2)
        quantity_best = np.round([i for i in indata_best['QUANTITY']][0],2)






        st.metric(label="Current price", value=pr, delta_color="off")
        st.metric(label="Optimum price", value=price_best, delta_color="off")
    with col2:
        
        
    
        st.metric(label="Expected demand on next week", value=[i for i in current_price['QUANTITY']][0],
        delta_color="off")
        st.metric(label="Optimum quantity", value=quantity_best,
        delta_color="off")
        
    with col3:
        st.metric(label="Next week profit based on current price", value=[i for i in current_price['PROFIT']][0],
        delta_color="off")
        st.metric(label="Next week profit based on optimum price", value=profit_best,
        delta_color="off")
        



    
    
    #############################

    
    

    A, B = st.columns(2)
    
    with A:
        
        st.write('**The trend below shows the change of Price over time**')
        
        TOOLS = 'save,pan,box_zoom,reset,wheel_zoom,hover'
        pq = figure(x_axis_type="datetime", title="Price over time", y_axis_type="linear", plot_height = 350,
                tools = TOOLS, plot_width = 600)
        pq.xaxis.axis_label = 'Time'
        pq.yaxis.axis_label = 'Price per item'
        # p.circle(2010, product_.regular_price.min(), size = 10, color = 'red')

        pq.line(product_.date_week, product_.regular_price,line_color="purple", line_width = 3)
        pq.select_one(HoverTool).tooltips = [
            ('Time', '@x'),
            ('Price per item', '@y'),
        ]

        output_file("reqular_price1.html", title="Line Chart")
        save(pq)
        p1 = open("reqular_price1.html", errors="ignore")
        components.html(p1.read(), height=500, width=700)
        
        
        
        
        # logged_model = 'runs:/a4e3a4228f674dcb865f949651c47bc2/models'

        # Load model as a PyFuncModel.
        # loaded_model = mlflow.pyfunc.load_model(logged_model)

    
        test = pd.DataFrame(columns= [ "PRICE", "QUANTITY"])
        
        # price in step of 0.01 and denominations increase in that sense
        test["PRICE"] = np.arange(minn-20, maxx, 0.01)
        test["QUANTITY"] = loaded_model.predict(pd.DataFrame(test['PRICE']))
        
        # Profit
        # 
        test['PROFIT'] = (test["PRICE"] - mean_price) * test['QUANTITY'] 
        
        
        ind = np.where(test['PROFIT'] == test['PROFIT'].max())[0][0]
        indata = test.loc[[ind]]
        
        
        price = [i for i in indata['PRICE']][0]
        profit = [i for i in indata['PROFIT']][0]
        
        
        
        
        st.write('**The graph shows exact price for maximum profit**')
        
        TOOLS = 'save,pan,box_zoom,reset,wheel_zoom,hover'
        pq = figure(title="Profit over Price", y_axis_type="linear", plot_height = 350,
                tools = TOOLS, plot_width = 600)
        pq.xaxis.axis_label = 'Price'
        pq.yaxis.axis_label = 'Profit'
        # p.circle(2010, product_.regular_price.min(), size = 10, color = 'red')
        # figure out the index point for max profit
        

        pq.line(test.PRICE, test.PROFIT,line_color="purple", line_width = 3)
        pq.select_one(HoverTool).tooltips = [
            ('Price', '@x'),
            ('Profit', '@y'),
        ]
        
     
        

        
        vline = Span(location=price, dimension='height', line_color='red', line_width=1)
        # Horizontal line
        hline = Span(location=profit, dimension='width', line_color='green', line_width=1)

        pq.renderers.extend([vline, hline])
        
        
        
    

        output_file("optimum_price.html", title="Line Chart")
        save(pq)
        p1 = open("optimum_price.html", errors="ignore")
        components.html(p1.read(), height=500, width=700)
        
       
        
        
    
    
    with B:
        
        st.write('**The trend below shows the change of quantity over time**')
        
        TOOLS = 'save,pan,box_zoom,reset,wheel_zoom,hover'
        p = figure(x_axis_type="datetime", title="Quantity over time", y_axis_type="linear", plot_height = 350,
                tools = TOOLS, plot_width = 600)
        p.xaxis.axis_label = 'Time'
        p.yaxis.axis_label = 'Quantity sold'
        # p.circle(2010, product_.regular_price.min(), size = 10, color = 'red')

        p.line(product_.date_week, product_.regular_volume,line_color="purple", line_width = 3)
        p.select_one(HoverTool).tooltips = [
            ('Time', '@x'),
            ('Number of item sold', '@y'),
        ]

        output_file("regular_volume1.html", title="Line Chart")
        save(p)   

        p2 = open("regular_volume1.html", errors="ignore")
        components.html(p2.read(), height=500, width=700)
 
 
 ##########################################
        st.write('**The trend below shows that product is very sensitive to price changes**')
 
        TOOLS = 'save,pan,box_zoom,reset,wheel_zoom,hover'
        paa = figure(title="Quantity over time", y_axis_type="linear", plot_height = 350,
                tools = TOOLS, plot_width = 600)
        paa.xaxis.axis_label = 'Price'
        paa.yaxis.axis_label = 'Quantity'
        # p.circle(2010, product_.regular_price.min(), size = 10, color = 'red')
        # logged_model = 'runs:/a4e3a4228f674dcb865f949651c47bc2/models'

        
        slope = loaded_model.coef_[0]
        intercept = loaded_model.intercept_

        # Make the regression line
        regression_line = Slope(gradient=slope, y_intercept=intercept, line_color="red")
        
        paa.add_layout(regression_line)
        

        paa.scatter(product_.regular_price, product_.regular_volume,line_color="purple", line_width = 3)
        paa.select_one(HoverTool).tooltips = [
            ('Time', '@x'),
            ('Number of item sold', '@y'),
        ]

        paa.add_layout(regression_line)

        output_file("quantity_price.html", title="Line Chart")
        save(paa)   

        paa = open("quantity_price.html", errors="ignore")
        components.html(paa.read(), height=500, width=700)
 
 
 
 
######################################################################### 
 

if option == 'Product_ID: 84630314':
    st.title('Product_ID: 84630314')
    
    loaded_model_prom  = joblib.load('Modelss/model_regular_84630314.pkl') 
    loaded_model = joblib.load('Modelss/model_promo_84630314.pkl') 
    
    Price_Elasticity_regular = np.round(loaded_model.coef_[0], 2)
    Price_Elasticity_promo = np.round(loaded_model_prom.coef_[0], 2)
    
    st.subheader(f'Price Elasticity for regular price: {Price_Elasticity_regular}')
    st.subheader(f'Price Elasticity for promo price: {Price_Elasticity_promo}')
    
    
    df = pd.read_csv('Senior Data Assignment.csv')
    
    columns = [i for i in df.columns]
    sku_cde = [i for i in df['sku_cde'].unique()]
    df_regular = df[columns[0:5]]
    df_regular = df_regular.dropna()
    
    
    
    # product_ = pd.read_csv('Senior Data Assignment.csv')
    product_ = df_regular[df_regular['sku_cde']==84630314]
    product_['date_week'] = pd.to_datetime(product_['date_week'])
    # product_['year'] = pd.to_datetime(product_['date_week']).dt.year
    
    # st.dataframe(product_)
    ##pr---- price
    
    maxx = [i for i in product_[['regular_price']].max()][0]
    minn = [i for i in product_[['regular_price']].min()][0]
    
    ### avarage cost price for the product
    mean_price = np.round(product_['cost_price'].mean(), 2)

    
    mid = (maxx + minn)/2
    
    with st.sidebar:
        
        pr = st.sidebar.slider('Choose price', minn, maxx+1, mid)
        st.write('For Product ID: 111708109, we see that max price was 186 when the optimal price could be 126.47 for maximum profit . this means the Product was loosing in profit. but if they had sold for 126 the sales may decrease, quantity been sold will decrease off course but profit will increase.')
    
        st.write('**Please note that all the products can be interpreted in an similar manner**')
    
        st.write('Now that we have these new price points for each items, once they are set up is important to continuously monitoring the sales and profits made out of these items.')


        st.write('**Elasticities interpretation**')
        
        st.write('A 10% price decrease in Product ID: 111708109 , it increases sales demand by 172.8% or a 10% price increase in Product ID: 62875832, it decreases sales demand by 172.8%')



    
    col1, col2, col3 = st.columns(3)
    # col1.metric("Temperature", "70 °F", "1.2 °F")
    # col2.metric("Wind", "9 mph", "-8%")
    # col3.metric("Humidity", "86%", "4%")
    
    #############################################################################################
    with col1:
        
        
        
      #####$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        # loaded_model = mlflow.pyfunc.load_model(logged_model)
        
        current_price = pd.DataFrame(columns= [ "PRICE", "QUANTITY"])
        current_price['PRICE'] = [pr]
        current_price['QUANTITY'] = np.round(loaded_model.predict(pd.DataFrame([pr])), 2)
        current_price['PROFIT'] = np.round((current_price["PRICE"] - mean_price) * current_price['QUANTITY'], 2)
        
        
        test = pd.DataFrame(columns= [ "PRICE", "QUANTITY"])
        
        # price in step of 0.01 and denominations increase in that sense
        test["PRICE"] = np.arange(minn, maxx, 0.01)
        test["QUANTITY"] = loaded_model.predict(pd.DataFrame(test['PRICE']))
        
        # Profit
        test['PROFIT'] = (test["PRICE"] - mean_price) * test['QUANTITY'] 
        
        ind = np.where(test['PROFIT'] == test['PROFIT'].max())[0][0]
        indata = test.loc[[ind]]
        
        
        pricee = np.round([i for i in indata['PRICE']][0],2)
        profitt = np.round([i for i in indata['PROFIT']][0], 2)
        quantity = np.round([i for i in indata['QUANTITY']][0], 2)
       
#########################################################################################



        test_best = pd.DataFrame(columns= [ "PRICE", "QUANTITY"])
        test_best["PRICE"] = np.arange(minn-20, maxx, 0.01)
        test_best["QUANTITY"] = loaded_model.predict(pd.DataFrame(test_best['PRICE']))
        
        # Profit
        # 
        test_best['PROFIT'] = (test_best["PRICE"] - mean_price) * test_best['QUANTITY'] 
        
        
        ind_best = np.where(test_best['PROFIT'] == test_best['PROFIT'].max())[0][0]
        indata_best = test_best.loc[[ind_best]]
        
        
        price_best = np.round([i for i in indata_best['PRICE']][0],2)
        profit_best = np.round([i for i in indata_best['PROFIT']][0],2)
        quantity_best = np.round([i for i in indata_best['QUANTITY']][0],2)






        st.metric(label="Current price", value=pr, delta_color="off")
        st.metric(label="Optimum price", value=price_best, delta_color="off")
    with col2:
        
        
    
        st.metric(label="Expected demand on next week", value=[i for i in current_price['QUANTITY']][0],
        delta_color="off")
        st.metric(label="Optimum quantity", value=quantity_best,
        delta_color="off")
        
    with col3:
        st.metric(label="Next week profit based on current price", value=[i for i in current_price['PROFIT']][0],
        delta_color="off")
        st.metric(label="Next week profit based on optimum price", value=profit_best,
        delta_color="off")
        



    
    
    #############################

    
    

    A, B = st.columns(2)
    
    with A:
        
        st.write('**The trend below shows the change of Price over time**')
        
        TOOLS = 'save,pan,box_zoom,reset,wheel_zoom,hover'
        pq = figure(x_axis_type="datetime", title="Price over time", y_axis_type="linear", plot_height = 350,
                tools = TOOLS, plot_width = 600)
        pq.xaxis.axis_label = 'Time'
        pq.yaxis.axis_label = 'Price per item'
        # p.circle(2010, product_.regular_price.min(), size = 10, color = 'red')

        pq.line(product_.date_week, product_.regular_price,line_color="purple", line_width = 3)
        pq.select_one(HoverTool).tooltips = [
            ('Time', '@x'),
            ('Price per item', '@y'),
        ]

        output_file("reqular_price1.html", title="Line Chart")
        save(pq)
        p1 = open("reqular_price1.html", errors="ignore")
        components.html(p1.read(), height=500, width=700)
        
        
        
        
        # logged_model = 'runs:/a4e3a4228f674dcb865f949651c47bc2/models'

        # Load model as a PyFuncModel.
        # loaded_model = mlflow.pyfunc.load_model(logged_model)

    
        test = pd.DataFrame(columns= [ "PRICE", "QUANTITY"])
        
        # price in step of 0.01 and denominations increase in that sense
        test["PRICE"] = np.arange(minn-20, maxx, 0.01)
        test["QUANTITY"] = loaded_model.predict(pd.DataFrame(test['PRICE']))
        
        # Profit
        # 
        test['PROFIT'] = (test["PRICE"] - mean_price) * test['QUANTITY'] 
        
        
        ind = np.where(test['PROFIT'] == test['PROFIT'].max())[0][0]
        indata = test.loc[[ind]]
        
        
        price = [i for i in indata['PRICE']][0]
        profit = [i for i in indata['PROFIT']][0]
        
        
        
        
        st.write('**The graph shows exact price for maximum profit**')
        
        TOOLS = 'save,pan,box_zoom,reset,wheel_zoom,hover'
        pq = figure(title="Profit over Price", y_axis_type="linear", plot_height = 350,
                tools = TOOLS, plot_width = 600)
        pq.xaxis.axis_label = 'Price'
        pq.yaxis.axis_label = 'Profit'
        # p.circle(2010, product_.regular_price.min(), size = 10, color = 'red')
        # figure out the index point for max profit
        

        pq.line(test.PRICE, test.PROFIT,line_color="purple", line_width = 3)
        pq.select_one(HoverTool).tooltips = [
            ('Price', '@x'),
            ('Profit', '@y'),
        ]
        
     
        

        
        vline = Span(location=price, dimension='height', line_color='red', line_width=1)
        # Horizontal line
        hline = Span(location=profit, dimension='width', line_color='green', line_width=1)

        pq.renderers.extend([vline, hline])
        
        
        
    

        output_file("optimum_price.html", title="Line Chart")
        save(pq)
        p1 = open("optimum_price.html", errors="ignore")
        components.html(p1.read(), height=500, width=700)
        
       
        
        
    
    
    with B:
        
        st.write('**The trend below shows the change of quantity over time**')
        
        TOOLS = 'save,pan,box_zoom,reset,wheel_zoom,hover'
        p = figure(x_axis_type="datetime", title="Quantity over time", y_axis_type="linear", plot_height = 350,
                tools = TOOLS, plot_width = 600)
        p.xaxis.axis_label = 'Time'
        p.yaxis.axis_label = 'Quantity sold'
        # p.circle(2010, product_.regular_price.min(), size = 10, color = 'red')

        p.line(product_.date_week, product_.regular_volume,line_color="purple", line_width = 3)
        p.select_one(HoverTool).tooltips = [
            ('Time', '@x'),
            ('Number of item sold', '@y'),
        ]

        output_file("regular_volume1.html", title="Line Chart")
        save(p)   

        p2 = open("regular_volume1.html", errors="ignore")
        components.html(p2.read(), height=500, width=700)
 
 
 ##########################################
        st.write('**The trend below shows that product is very sensitive to price changes**')
 
        TOOLS = 'save,pan,box_zoom,reset,wheel_zoom,hover'
        paa = figure(title="Quantity over time", y_axis_type="linear", plot_height = 350,
                tools = TOOLS, plot_width = 600)
        paa.xaxis.axis_label = 'Price'
        paa.yaxis.axis_label = 'Quantity'
        # p.circle(2010, product_.regular_price.min(), size = 10, color = 'red')
        # logged_model = 'runs:/a4e3a4228f674dcb865f949651c47bc2/models'

        
        slope = loaded_model.coef_[0]
        intercept = loaded_model.intercept_

        # Make the regression line
        regression_line = Slope(gradient=slope, y_intercept=intercept, line_color="red")
        
        paa.add_layout(regression_line)
        

        paa.scatter(product_.regular_price, product_.regular_volume,line_color="purple", line_width = 3)
        paa.select_one(HoverTool).tooltips = [
            ('Time', '@x'),
            ('Number of item sold', '@y'),
        ]

        paa.add_layout(regression_line)

        output_file("quantity_price.html", title="Line Chart")
        save(paa)   

        paa = open("quantity_price.html", errors="ignore")
        components.html(paa.read(), height=500, width=700)






if option == 'Product_ID: 95208654':
    st.title('Product_ID: 95208654')
    
    loaded_model_prom  = joblib.load('Modelss/model_regular_95208654.pkl') 
    loaded_model = joblib.load('Modelss/model_promo_95208654.pkl') 
    
    Price_Elasticity_regular = np.round(loaded_model.coef_[0], 2)
    Price_Elasticity_promo = np.round(loaded_model_prom.coef_[0], 2)
    
    st.subheader(f'Price Elasticity for regular price: {Price_Elasticity_regular}')
    st.subheader(f'Price Elasticity for promo price: {Price_Elasticity_promo}')
    
    
    df = pd.read_csv('Senior Data Assignment.csv')
    
    columns = [i for i in df.columns]
    sku_cde = [i for i in df['sku_cde'].unique()]
    df_regular = df[columns[0:5]]
    df_regular = df_regular.dropna()
    
    
    
    # product_ = pd.read_csv('Senior Data Assignment.csv')
    product_ = df_regular[df_regular['sku_cde']==95208654]
    product_['date_week'] = pd.to_datetime(product_['date_week'])
    # product_['year'] = pd.to_datetime(product_['date_week']).dt.year
    
    # st.dataframe(product_)
    ##pr---- price
    
    maxx = [i for i in product_[['regular_price']].max()][0]
    minn = [i for i in product_[['regular_price']].min()][0]
    
    ### avarage cost price for the product
    mean_price = np.round(product_['cost_price'].mean(), 2)

    
    mid = (maxx + minn)/2
    
    with st.sidebar:
        
        pr = st.sidebar.slider('Choose price', minn, maxx+1, mid)
        st.write('For Product ID: 111708109, we see that max price was 186 when the optimal price could be 126.47 for maximum profit . this means the Product was loosing in profit. but if they had sold for 126 the sales may decrease, quantity been sold will decrease off course but profit will increase.')
    
        st.write('**Please note that all the products can be interpreted in an similar manner**')
    
        st.write('Now that we have these new price points for each items, once they are set up is important to continuously monitoring the sales and profits made out of these items.')


        st.write('**Elasticities interpretation**')
        
        st.write('A 10% price decrease in Product ID: 111708109 , it increases sales demand by 172.8% or a 10% price increase in Product ID: 62875832, it decreases sales demand by 172.8%')



    
    col1, col2, col3 = st.columns(3)
    # col1.metric("Temperature", "70 °F", "1.2 °F")
    # col2.metric("Wind", "9 mph", "-8%")
    # col3.metric("Humidity", "86%", "4%")
    
    #############################################################################################
    with col1:
        
        
        
      #####$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        # loaded_model = mlflow.pyfunc.load_model(logged_model)
        
        current_price = pd.DataFrame(columns= [ "PRICE", "QUANTITY"])
        current_price['PRICE'] = [pr]
        current_price['QUANTITY'] = np.round(loaded_model.predict(pd.DataFrame([pr])), 2)
        current_price['PROFIT'] = np.round((current_price["PRICE"] - mean_price) * current_price['QUANTITY'], 2)
        
        
        test = pd.DataFrame(columns= [ "PRICE", "QUANTITY"])
        
        # price in step of 0.01 and denominations increase in that sense
        test["PRICE"] = np.arange(minn, maxx, 0.01)
        test["QUANTITY"] = loaded_model.predict(pd.DataFrame(test['PRICE']))
        
        # Profit
        test['PROFIT'] = (test["PRICE"] - mean_price) * test['QUANTITY'] 
        
        ind = np.where(test['PROFIT'] == test['PROFIT'].max())[0][0]
        indata = test.loc[[ind]]
        
        
        pricee = np.round([i for i in indata['PRICE']][0],2)
        profitt = np.round([i for i in indata['PROFIT']][0], 2)
        quantity = np.round([i for i in indata['QUANTITY']][0], 2)
       
#########################################################################################



        test_best = pd.DataFrame(columns= [ "PRICE", "QUANTITY"])
        test_best["PRICE"] = np.arange(minn-20, maxx, 0.01)
        test_best["QUANTITY"] = loaded_model.predict(pd.DataFrame(test_best['PRICE']))
        
        # Profit
        # 
        test_best['PROFIT'] = (test_best["PRICE"] - mean_price) * test_best['QUANTITY'] 
        
        
        ind_best = np.where(test_best['PROFIT'] == test_best['PROFIT'].max())[0][0]
        indata_best = test_best.loc[[ind_best]]
        
        
        price_best = np.round([i for i in indata_best['PRICE']][0],2)
        profit_best = np.round([i for i in indata_best['PROFIT']][0],2)
        quantity_best = np.round([i for i in indata_best['QUANTITY']][0],2)






        st.metric(label="Current price", value=pr, delta_color="off")
        st.metric(label="Optimum price", value=price_best, delta_color="off")
    with col2:
        
        
    
        st.metric(label="Expected demand on next week", value=[i for i in current_price['QUANTITY']][0],
        delta_color="off")
        st.metric(label="Optimum quantity", value=quantity_best,
        delta_color="off")
        
    with col3:
        st.metric(label="Next week profit based on current price", value=[i for i in current_price['PROFIT']][0],
        delta_color="off")
        st.metric(label="Next week profit based on optimum price", value=profit_best,
        delta_color="off")
        



    
    
    #############################

    
    

    A, B = st.columns(2)
    
    with A:
        
        st.write('**The trend below shows the change of Price over time**')
        
        TOOLS = 'save,pan,box_zoom,reset,wheel_zoom,hover'
        pq = figure(x_axis_type="datetime", title="Price over time", y_axis_type="linear", plot_height = 350,
                tools = TOOLS, plot_width = 600)
        pq.xaxis.axis_label = 'Time'
        pq.yaxis.axis_label = 'Price per item'
        # p.circle(2010, product_.regular_price.min(), size = 10, color = 'red')

        pq.line(product_.date_week, product_.regular_price,line_color="purple", line_width = 3)
        pq.select_one(HoverTool).tooltips = [
            ('Time', '@x'),
            ('Price per item', '@y'),
        ]

        output_file("reqular_price1.html", title="Line Chart")
        save(pq)
        p1 = open("reqular_price1.html", errors="ignore")
        components.html(p1.read(), height=500, width=700)
        
        
        
        
        # logged_model = 'runs:/a4e3a4228f674dcb865f949651c47bc2/models'

        # Load model as a PyFuncModel.
        # loaded_model = mlflow.pyfunc.load_model(logged_model)

    
        test = pd.DataFrame(columns= [ "PRICE", "QUANTITY"])
        
        # price in step of 0.01 and denominations increase in that sense
        test["PRICE"] = np.arange(minn-20, maxx, 0.01)
        test["QUANTITY"] = loaded_model.predict(pd.DataFrame(test['PRICE']))
        
        # Profit
        # 
        test['PROFIT'] = (test["PRICE"] - mean_price) * test['QUANTITY'] 
        
        
        ind = np.where(test['PROFIT'] == test['PROFIT'].max())[0][0]
        indata = test.loc[[ind]]
        
        
        price = [i for i in indata['PRICE']][0]
        profit = [i for i in indata['PROFIT']][0]
        
        
        
        
        st.write('**The graph shows exact price for maximum profit**')
        
        TOOLS = 'save,pan,box_zoom,reset,wheel_zoom,hover'
        pq = figure(title="Profit over Price", y_axis_type="linear", plot_height = 350,
                tools = TOOLS, plot_width = 600)
        pq.xaxis.axis_label = 'Price'
        pq.yaxis.axis_label = 'Profit'
        # p.circle(2010, product_.regular_price.min(), size = 10, color = 'red')
        # figure out the index point for max profit
        

        pq.line(test.PRICE, test.PROFIT,line_color="purple", line_width = 3)
        pq.select_one(HoverTool).tooltips = [
            ('Price', '@x'),
            ('Profit', '@y'),
        ]
        
     
        

        
        vline = Span(location=price, dimension='height', line_color='red', line_width=1)
        # Horizontal line
        hline = Span(location=profit, dimension='width', line_color='green', line_width=1)

        pq.renderers.extend([vline, hline])
        
        
        
    

        output_file("optimum_price.html", title="Line Chart")
        save(pq)
        p1 = open("optimum_price.html", errors="ignore")
        components.html(p1.read(), height=500, width=700)
        
       
        
        
    
    
    with B:
        
        st.write('**The trend below shows the change of quantity over time**')
        
        TOOLS = 'save,pan,box_zoom,reset,wheel_zoom,hover'
        p = figure(x_axis_type="datetime", title="Quantity over time", y_axis_type="linear", plot_height = 350,
                tools = TOOLS, plot_width = 600)
        p.xaxis.axis_label = 'Time'
        p.yaxis.axis_label = 'Quantity sold'
        # p.circle(2010, product_.regular_price.min(), size = 10, color = 'red')

        p.line(product_.date_week, product_.regular_volume,line_color="purple", line_width = 3)
        p.select_one(HoverTool).tooltips = [
            ('Time', '@x'),
            ('Number of item sold', '@y'),
        ]

        output_file("regular_volume1.html", title="Line Chart")
        save(p)   

        p2 = open("regular_volume1.html", errors="ignore")
        components.html(p2.read(), height=500, width=700)
 
 
 ##########################################
        st.write('**The trend below shows that product is very sensitive to price changes**')
 
        TOOLS = 'save,pan,box_zoom,reset,wheel_zoom,hover'
        paa = figure(title="Quantity over time", y_axis_type="linear", plot_height = 350,
                tools = TOOLS, plot_width = 600)
        paa.xaxis.axis_label = 'Price'
        paa.yaxis.axis_label = 'Quantity'
        # p.circle(2010, product_.regular_price.min(), size = 10, color = 'red')
        # logged_model = 'runs:/a4e3a4228f674dcb865f949651c47bc2/models'

        
        slope = loaded_model.coef_[0]
        intercept = loaded_model.intercept_

        # Make the regression line
        regression_line = Slope(gradient=slope, y_intercept=intercept, line_color="red")
        
        paa.add_layout(regression_line)
        

        paa.scatter(product_.regular_price, product_.regular_volume,line_color="purple", line_width = 3)
        paa.select_one(HoverTool).tooltips = [
            ('Time', '@x'),
            ('Number of item sold', '@y'),
        ]

        paa.add_layout(regression_line)

        output_file("quantity_price.html", title="Line Chart")
        save(paa)   

        paa = open("quantity_price.html", errors="ignore")
        components.html(paa.read(), height=500, width=700)





if option == 'Product_ID: 111708109':
    st.title('Product_ID: 111708109')
    
    loaded_model_prom  = joblib.load('Modelss/model_regular_111708109.pkl') 
    loaded_model = joblib.load('Modelss/model_promo_111708109.pkl') 
    
    Price_Elasticity_regular = np.round(loaded_model.coef_[0], 2)
    Price_Elasticity_promo = np.round(loaded_model_prom.coef_[0], 2)
    
    st.subheader(f'Price Elasticity for regular price: {Price_Elasticity_regular}')
    st.subheader(f'Price Elasticity for promo price: {Price_Elasticity_promo}')
    
    
    df = pd.read_csv('Senior Data Assignment.csv')
    
    columns = [i for i in df.columns]
    sku_cde = [i for i in df['sku_cde'].unique()]
    df_regular = df[columns[0:5]]
    df_regular = df_regular.dropna()
    
    
    
    # product_ = pd.read_csv('Senior Data Assignment.csv')
    product_ = df_regular[df_regular['sku_cde']==111708109]
    product_['date_week'] = pd.to_datetime(product_['date_week'])
    # product_['year'] = pd.to_datetime(product_['date_week']).dt.year
    
    # st.dataframe(product_)
    ##pr---- price
    
    maxx = [i for i in product_[['regular_price']].max()][0]
    minn = [i for i in product_[['regular_price']].min()][0]
    
    ### avarage cost price for the product
    mean_price = np.round(product_['cost_price'].mean(), 2)

    
    mid = (maxx + minn)/2
    
    with st.sidebar:
        
        pr = st.sidebar.slider('Choose price', minn, maxx+1, mid)
        st.write('For Product ID: 111708109, we see that max price was 186 when the optimal price could be 126.47 for maximum profit . this means the Product was loosing in profit. but if they had sold for 126 the sales may decrease, quantity been sold will decrease off course but profit will increase.')
    
        st.write('**Please note that all the products can be interpreted in an similar manner**')
    
        st.write('Now that we have these new price points for each items, once they are set up is important to continuously monitoring the sales and profits made out of these items.')


        st.write('**Elasticities interpretation**')
        
        st.write('A 10% price decrease in Product ID: 111708109 , it increases sales demand by 172.8% or a 10% price increase in Product ID: 62875832, it decreases sales demand by 172.8%')


    col1, col2, col3 = st.columns(3)
    # col1.metric("Temperature", "70 °F", "1.2 °F")
    # col2.metric("Wind", "9 mph", "-8%")
    # col3.metric("Humidity", "86%", "4%")
    
    #############################################################################################
    with col1:
        
        
        
      #####$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        # loaded_model = mlflow.pyfunc.load_model(logged_model)
        
        current_price = pd.DataFrame(columns= [ "PRICE", "QUANTITY"])
        current_price['PRICE'] = [pr]
        current_price['QUANTITY'] = np.round(loaded_model.predict(pd.DataFrame([pr])), 2)
        current_price['PROFIT'] = np.round((current_price["PRICE"] - mean_price) * current_price['QUANTITY'], 2)
        
        
        test = pd.DataFrame(columns= [ "PRICE", "QUANTITY"])
        
        # price in step of 0.01 and denominations increase in that sense
        test["PRICE"] = np.arange(minn, maxx, 0.01)
        test["QUANTITY"] = loaded_model.predict(pd.DataFrame(test['PRICE']))
        
        # Profit
        test['PROFIT'] = (test["PRICE"] - mean_price) * test['QUANTITY'] 
        
        ind = np.where(test['PROFIT'] == test['PROFIT'].max())[0][0]
        indata = test.loc[[ind]]
        
        
        pricee = np.round([i for i in indata['PRICE']][0],2)
        profitt = np.round([i for i in indata['PROFIT']][0], 2)
        quantity = np.round([i for i in indata['QUANTITY']][0], 2)
       
#########################################################################################



        test_best = pd.DataFrame(columns= [ "PRICE", "QUANTITY"])
        test_best["PRICE"] = np.arange(minn-20, maxx, 0.01)
        test_best["QUANTITY"] = loaded_model.predict(pd.DataFrame(test_best['PRICE']))
        
        # Profit
        # 
        test_best['PROFIT'] = (test_best["PRICE"] - mean_price) * test_best['QUANTITY'] 
        
        
        ind_best = np.where(test_best['PROFIT'] == test_best['PROFIT'].max())[0][0]
        indata_best = test_best.loc[[ind_best]]
        
        
        price_best = np.round([i for i in indata_best['PRICE']][0],2)
        profit_best = np.round([i for i in indata_best['PROFIT']][0],2)
        quantity_best = np.round([i for i in indata_best['QUANTITY']][0],2)






        st.metric(label="Current price", value=pr, delta_color="off")
        st.metric(label="Optimum price", value=price_best, delta_color="off")
    with col2:
        
        
    
        st.metric(label="Expected demand on next week", value=[i for i in current_price['QUANTITY']][0],
        delta_color="off")
        st.metric(label="Optimum quantity", value=quantity_best,
        delta_color="off")
        
    with col3:
        st.metric(label="Next week profit based on current price", value=[i for i in current_price['PROFIT']][0],
        delta_color="off")
        st.metric(label="Next week profit based on optimum price", value=profit_best,
        delta_color="off")
        



    
    
    #############################

    
    

    A, B = st.columns(2)
    
    with A:
        
        st.write('**The trend below shows the change of Price over time**')
        
        TOOLS = 'save,pan,box_zoom,reset,wheel_zoom,hover'
        pq = figure(x_axis_type="datetime", title="Price over time", y_axis_type="linear", plot_height = 350,
                tools = TOOLS, plot_width = 600)
        pq.xaxis.axis_label = 'Time'
        pq.yaxis.axis_label = 'Price per item'
        # p.circle(2010, product_.regular_price.min(), size = 10, color = 'red')

        pq.line(product_.date_week, product_.regular_price,line_color="purple", line_width = 3)
        pq.select_one(HoverTool).tooltips = [
            ('Time', '@x'),
            ('Price per item', '@y'),
        ]

        output_file("reqular_price1.html", title="Line Chart")
        save(pq)
        p1 = open("reqular_price1.html", errors="ignore")
        components.html(p1.read(), height=500, width=700)
        
        
        
        
        # logged_model = 'runs:/a4e3a4228f674dcb865f949651c47bc2/models'

        # Load model as a PyFuncModel.
        # loaded_model = mlflow.pyfunc.load_model(logged_model)

    
        test = pd.DataFrame(columns= [ "PRICE", "QUANTITY"])
        
        # price in step of 0.01 and denominations increase in that sense
        test["PRICE"] = np.arange(minn-20, maxx, 0.01)
        test["QUANTITY"] = loaded_model.predict(pd.DataFrame(test['PRICE']))
        
        # Profit
        # 
        test['PROFIT'] = (test["PRICE"] - mean_price) * test['QUANTITY'] 
        
        
        ind = np.where(test['PROFIT'] == test['PROFIT'].max())[0][0]
        indata = test.loc[[ind]]
        
        
        price = [i for i in indata['PRICE']][0]
        profit = [i for i in indata['PROFIT']][0]
        
        
        
        
        st.write('**The graph shows exact price for maximum profit**')
        
        TOOLS = 'save,pan,box_zoom,reset,wheel_zoom,hover'
        pq = figure(title="Profit over Price", y_axis_type="linear", plot_height = 350,
                tools = TOOLS, plot_width = 600)
        pq.xaxis.axis_label = 'Price'
        pq.yaxis.axis_label = 'Profit'
        # p.circle(2010, product_.regular_price.min(), size = 10, color = 'red')
        # figure out the index point for max profit
        

        pq.line(test.PRICE, test.PROFIT,line_color="purple", line_width = 3)
        pq.select_one(HoverTool).tooltips = [
            ('Price', '@x'),
            ('Profit', '@y'),
        ]
        
     
        

        
        vline = Span(location=price, dimension='height', line_color='red', line_width=1)
        # Horizontal line
        hline = Span(location=profit, dimension='width', line_color='green', line_width=1)

        pq.renderers.extend([vline, hline])
        
        
        
    

        output_file("optimum_price.html", title="Line Chart")
        save(pq)
        p1 = open("optimum_price.html", errors="ignore")
        components.html(p1.read(), height=500, width=700)
        
       
        
        
    
    
    with B:
        
        st.write('**The trend below shows the change of quantity over time**')
        
        TOOLS = 'save,pan,box_zoom,reset,wheel_zoom,hover'
        p = figure(x_axis_type="datetime", title="Quantity over time", y_axis_type="linear", plot_height = 350,
                tools = TOOLS, plot_width = 600)
        p.xaxis.axis_label = 'Time'
        p.yaxis.axis_label = 'Quantity sold'
        # p.circle(2010, product_.regular_price.min(), size = 10, color = 'red')

        p.line(product_.date_week, product_.regular_volume,line_color="purple", line_width = 3)
        p.select_one(HoverTool).tooltips = [
            ('Time', '@x'),
            ('Number of item sold', '@y'),
        ]

        output_file("regular_volume1.html", title="Line Chart")
        save(p)   

        p2 = open("regular_volume1.html", errors="ignore")
        components.html(p2.read(), height=500, width=700)
 
 
 ##########################################
        st.write('**The trend below shows that product is very sensitive to price changes**')
 
        TOOLS = 'save,pan,box_zoom,reset,wheel_zoom,hover'
        paa = figure(title="Quantity over time", y_axis_type="linear", plot_height = 350,
                tools = TOOLS, plot_width = 600)
        paa.xaxis.axis_label = 'Price'
        paa.yaxis.axis_label = 'Quantity'
        # p.circle(2010, product_.regular_price.min(), size = 10, color = 'red')
        # logged_model = 'runs:/a4e3a4228f674dcb865f949651c47bc2/models'

        
        slope = loaded_model.coef_[0]
        intercept = loaded_model.intercept_

        # Make the regression line
        regression_line = Slope(gradient=slope, y_intercept=intercept, line_color="red")
        
        paa.add_layout(regression_line)
        

        paa.scatter(product_.regular_price, product_.regular_volume,line_color="purple", line_width = 3)
        paa.select_one(HoverTool).tooltips = [
            ('Time', '@x'),
            ('Number of item sold', '@y'),
        ]

        paa.add_layout(regression_line)

        output_file("quantity_price.html", title="Line Chart")
        save(paa)   

        paa = open("quantity_price.html", errors="ignore")
        components.html(paa.read(), height=500, width=700)





if option == 'Product_ID: 73267284':
    st.title('Product_ID: 73267284')
    
    loaded_model_prom  = joblib.load('Modelss/model_regular_73267284.pkl') 
    loaded_model = joblib.load('Modelss/model_promo_73267284.pkl') 
    
    Price_Elasticity_regular = np.round(loaded_model.coef_[0], 2)
    Price_Elasticity_promo = np.round(loaded_model_prom.coef_[0], 2)
    
    st.subheader(f'Price Elasticity for regular price: {Price_Elasticity_regular}')
    st.subheader(f'Price Elasticity for promo price: {Price_Elasticity_promo}')
    
    
    df = pd.read_csv('Senior Data Assignment.csv')
    
    columns = [i for i in df.columns]
    sku_cde = [i for i in df['sku_cde'].unique()]
    df_regular = df[columns[0:5]]
    df_regular = df_regular.dropna()
    
    
    
    # product_ = pd.read_csv('Senior Data Assignment.csv')
    product_ = df_regular[df_regular['sku_cde']==73267284]
    product_['date_week'] = pd.to_datetime(product_['date_week'])
    # product_['year'] = pd.to_datetime(product_['date_week']).dt.year
    
    # st.dataframe(product_)
    ##pr---- price
    
    maxx = [i for i in product_[['regular_price']].max()][0]
    minn = [i for i in product_[['regular_price']].min()][0]
    
    ### avarage cost price for the product
    mean_price = np.round(product_['cost_price'].mean(), 2)

    
    mid = (maxx + minn)/2
    
    with st.sidebar:
        
        pr = st.sidebar.slider('Choose price', minn, maxx+1, mid)
        st.write('For Product ID: 111708109, we see that max price was 186 when the optimal price could be 126.47 for maximum profit . this means the Product was loosing in profit. but if they had sold for 126 the sales may decrease, quantity been sold will decrease off course but profit will increase.')
    
        st.write('**Please note that all the products can be interpreted in an similar manner**')
    
        st.write('Now that we have these new price points for each items, once they are set up is important to continuously monitoring the sales and profits made out of these items.')


        st.write('**Elasticities interpretation**')
        
        st.write('A 10% price decrease in Product ID: 111708109 , it increases sales demand by 172.8% or a 10% price increase in Product ID: 62875832, it decreases sales demand by 172.8%')



    col1, col2, col3 = st.columns(3)
    # col1.metric("Temperature", "70 °F", "1.2 °F")
    # col2.metric("Wind", "9 mph", "-8%")
    # col3.metric("Humidity", "86%", "4%")
    
    #############################################################################################
    with col1:
        
        
        
      #####$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        # loaded_model = mlflow.pyfunc.load_model(logged_model)
        
        current_price = pd.DataFrame(columns= [ "PRICE", "QUANTITY"])
        current_price['PRICE'] = [pr]
        current_price['QUANTITY'] = np.round(loaded_model.predict(pd.DataFrame([pr])), 2)
        current_price['PROFIT'] = np.round((current_price["PRICE"] - mean_price) * current_price['QUANTITY'], 2)
        
        
        test = pd.DataFrame(columns= [ "PRICE", "QUANTITY"])
        
        # price in step of 0.01 and denominations increase in that sense
        test["PRICE"] = np.arange(minn, maxx, 0.01)
        test["QUANTITY"] = loaded_model.predict(pd.DataFrame(test['PRICE']))
        
        # Profit
        test['PROFIT'] = (test["PRICE"] - mean_price) * test['QUANTITY'] 
        
        ind = np.where(test['PROFIT'] == test['PROFIT'].max())[0][0]
        indata = test.loc[[ind]]
        
        
        pricee = np.round([i for i in indata['PRICE']][0],2)
        profitt = np.round([i for i in indata['PROFIT']][0], 2)
        quantity = np.round([i for i in indata['QUANTITY']][0], 2)
       
#########################################################################################



        test_best = pd.DataFrame(columns= [ "PRICE", "QUANTITY"])
        test_best["PRICE"] = np.arange(minn-20, maxx, 0.01)
        test_best["QUANTITY"] = loaded_model.predict(pd.DataFrame(test_best['PRICE']))
        
        # Profit
        # 
        test_best['PROFIT'] = (test_best["PRICE"] - mean_price) * test_best['QUANTITY'] 
        
        
        ind_best = np.where(test_best['PROFIT'] == test_best['PROFIT'].max())[0][0]
        indata_best = test_best.loc[[ind_best]]
        
        
        price_best = np.round([i for i in indata_best['PRICE']][0],2)
        profit_best = np.round([i for i in indata_best['PROFIT']][0],2)
        quantity_best = np.round([i for i in indata_best['QUANTITY']][0],2)






        st.metric(label="Current price", value=pr, delta_color="off")
        st.metric(label="Optimum price", value=price_best, delta_color="off")
    with col2:
        
        
    
        st.metric(label="Expected demand on next week", value=[i for i in current_price['QUANTITY']][0],
        delta_color="off")
        st.metric(label="Optimum quantity", value=quantity_best,
        delta_color="off")
        
    with col3:
        st.metric(label="Next week profit based on current price", value=[i for i in current_price['PROFIT']][0],
        delta_color="off")
        st.metric(label="Next week profit based on optimum price", value=profit_best,
        delta_color="off")
        



    
    
    #############################

    
    

    A, B = st.columns(2)
    
    with A:
        
        st.write('**The trend below shows the change of Price over time**')
        
        TOOLS = 'save,pan,box_zoom,reset,wheel_zoom,hover'
        pq = figure(x_axis_type="datetime", title="Price over time", y_axis_type="linear", plot_height = 350,
                tools = TOOLS, plot_width = 600)
        pq.xaxis.axis_label = 'Time'
        pq.yaxis.axis_label = 'Price per item'
        # p.circle(2010, product_.regular_price.min(), size = 10, color = 'red')

        pq.line(product_.date_week, product_.regular_price,line_color="purple", line_width = 3)
        pq.select_one(HoverTool).tooltips = [
            ('Time', '@x'),
            ('Price per item', '@y'),
        ]

        output_file("reqular_price1.html", title="Line Chart")
        save(pq)
        p1 = open("reqular_price1.html", errors="ignore")
        components.html(p1.read(), height=500, width=700)
        
        
        
        
        # logged_model = 'runs:/a4e3a4228f674dcb865f949651c47bc2/models'

        # Load model as a PyFuncModel.
        # loaded_model = mlflow.pyfunc.load_model(logged_model)

    
        test = pd.DataFrame(columns= [ "PRICE", "QUANTITY"])
        
        # price in step of 0.01 and denominations increase in that sense
        test["PRICE"] = np.arange(minn-20, maxx, 0.01)
        test["QUANTITY"] = loaded_model.predict(pd.DataFrame(test['PRICE']))
        
        # Profit
        # 
        test['PROFIT'] = (test["PRICE"] - mean_price) * test['QUANTITY'] 
        
        
        ind = np.where(test['PROFIT'] == test['PROFIT'].max())[0][0]
        indata = test.loc[[ind]]
        
        
        price = [i for i in indata['PRICE']][0]
        profit = [i for i in indata['PROFIT']][0]
        
        
        
        
        st.write('**The graph shows exact price for maximum profit**')
        
        TOOLS = 'save,pan,box_zoom,reset,wheel_zoom,hover'
        pq = figure(title="Profit over Price", y_axis_type="linear", plot_height = 350,
                tools = TOOLS, plot_width = 600)
        pq.xaxis.axis_label = 'Price'
        pq.yaxis.axis_label = 'Profit'
        # p.circle(2010, product_.regular_price.min(), size = 10, color = 'red')
        # figure out the index point for max profit
        

        pq.line(test.PRICE, test.PROFIT,line_color="purple", line_width = 3)
        pq.select_one(HoverTool).tooltips = [
            ('Price', '@x'),
            ('Profit', '@y'),
        ]
        
     
        

        
        vline = Span(location=price, dimension='height', line_color='red', line_width=1)
        # Horizontal line
        hline = Span(location=profit, dimension='width', line_color='green', line_width=1)

        pq.renderers.extend([vline, hline])
        
        
        
    

        output_file("optimum_price.html", title="Line Chart")
        save(pq)
        p1 = open("optimum_price.html", errors="ignore")
        components.html(p1.read(), height=500, width=700)
        
       
        
        
    
    
    with B:
        
        st.write('**The trend below shows the change of quantity over time**')
        
        TOOLS = 'save,pan,box_zoom,reset,wheel_zoom,hover'
        p = figure(x_axis_type="datetime", title="Quantity over time", y_axis_type="linear", plot_height = 350,
                tools = TOOLS, plot_width = 600)
        p.xaxis.axis_label = 'Time'
        p.yaxis.axis_label = 'Quantity sold'
        # p.circle(2010, product_.regular_price.min(), size = 10, color = 'red')

        p.line(product_.date_week, product_.regular_volume,line_color="purple", line_width = 3)
        p.select_one(HoverTool).tooltips = [
            ('Time', '@x'),
            ('Number of item sold', '@y'),
        ]

        output_file("regular_volume1.html", title="Line Chart")
        save(p)   

        p2 = open("regular_volume1.html", errors="ignore")
        components.html(p2.read(), height=500, width=700)
 
 
 ##########################################
        st.write('**The trend below shows that product is very sensitive to price changes**')
 
        TOOLS = 'save,pan,box_zoom,reset,wheel_zoom,hover'
        paa = figure(title="Quantity over time", y_axis_type="linear", plot_height = 350,
                tools = TOOLS, plot_width = 600)
        paa.xaxis.axis_label = 'Price'
        paa.yaxis.axis_label = 'Quantity'
        # p.circle(2010, product_.regular_price.min(), size = 10, color = 'red')
        # logged_model = 'runs:/a4e3a4228f674dcb865f949651c47bc2/models'

        
        slope = loaded_model.coef_[0]
        intercept = loaded_model.intercept_

        # Make the regression line
        regression_line = Slope(gradient=slope, y_intercept=intercept, line_color="red")
        
        paa.add_layout(regression_line)
        

        paa.scatter(product_.regular_price, product_.regular_volume,line_color="purple", line_width = 3)
        paa.select_one(HoverTool).tooltips = [
            ('Time', '@x'),
            ('Number of item sold', '@y'),
        ]

        paa.add_layout(regression_line)

        output_file("quantity_price.html", title="Line Chart")
        save(paa)   

        paa = open("quantity_price.html", errors="ignore")
        components.html(paa.read(), height=500, width=700)





if option == 'Product_ID: 11990782':
    st.title('Product_ID: 11990782')
    
    loaded_model_prom  = joblib.load('Modelss/model_regular_11990782.pkl') 
    loaded_model = joblib.load('Modelss/model_promo_11990782.pkl') 
    
    Price_Elasticity_regular = np.round(loaded_model.coef_[0], 2)
    Price_Elasticity_promo = np.round(loaded_model_prom.coef_[0], 2)
    
    st.subheader(f'Price Elasticity for regular price: {Price_Elasticity_regular}')
    st.subheader(f'Price Elasticity for promo price: {Price_Elasticity_promo}')
    
    
    df = pd.read_csv('Senior Data Assignment.csv')
    
    columns = [i for i in df.columns]
    sku_cde = [i for i in df['sku_cde'].unique()]
    df_regular = df[columns[0:5]]
    df_regular = df_regular.dropna()
    
    
    
    # product_ = pd.read_csv('Senior Data Assignment.csv')
    product_ = df_regular[df_regular['sku_cde']==11990782]
    product_['date_week'] = pd.to_datetime(product_['date_week'])
    # product_['year'] = pd.to_datetime(product_['date_week']).dt.year
    
    # st.dataframe(product_)
    ##pr---- price
    
    maxx = [i for i in product_[['regular_price']].max()][0]
    minn = [i for i in product_[['regular_price']].min()][0]
    
    ### avarage cost price for the product
    mean_price = np.round(product_['cost_price'].mean(), 2)

    
    mid = (maxx + minn)/2
    
    with st.sidebar:
        
        pr = st.sidebar.slider('Choose price', minn, maxx+1, mid)
        st.write('For Product ID: 111708109, we see that max price was 186 when the optimal price could be 126.47 for maximum profit . this means the Product was loosing in profit. but if they had sold for 126 the sales may decrease, quantity been sold will decrease off course but profit will increase.')
    
        st.write('**Please note that all the products can be interpreted in an similar manner**')
    
        st.write('Now that we have these new price points for each items, once they are set up is important to continuously monitoring the sales and profits made out of these items.')


        st.write('**Elasticities interpretation**')
        
        st.write('A 10% price decrease in Product ID: 111708109 , it increases sales demand by 172.8% or a 10% price increase in Product ID: 62875832, it decreases sales demand by 172.8%')



    col1, col2, col3 = st.columns(3)
    # col1.metric("Temperature", "70 °F", "1.2 °F")
    # col2.metric("Wind", "9 mph", "-8%")
    # col3.metric("Humidity", "86%", "4%")
    
    #############################################################################################
    with col1:
        
        
        
      #####$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        # loaded_model = mlflow.pyfunc.load_model(logged_model)
        
        current_price = pd.DataFrame(columns= [ "PRICE", "QUANTITY"])
        current_price['PRICE'] = [pr]
        current_price['QUANTITY'] = np.round(loaded_model.predict(pd.DataFrame([pr])), 2)
        current_price['PROFIT'] = np.round((current_price["PRICE"] - mean_price) * current_price['QUANTITY'], 2)
        
        
        test = pd.DataFrame(columns= [ "PRICE", "QUANTITY"])
        
        # price in step of 0.01 and denominations increase in that sense
        test["PRICE"] = np.arange(minn, maxx, 0.01)
        test["QUANTITY"] = loaded_model.predict(pd.DataFrame(test['PRICE']))
        
        # Profit
        test['PROFIT'] = (test["PRICE"] - mean_price) * test['QUANTITY'] 
        
        ind = np.where(test['PROFIT'] == test['PROFIT'].max())[0][0]
        indata = test.loc[[ind]]
        
        
        pricee = np.round([i for i in indata['PRICE']][0],2)
        profitt = np.round([i for i in indata['PROFIT']][0], 2)
        quantity = np.round([i for i in indata['QUANTITY']][0], 2)
       
#########################################################################################



        test_best = pd.DataFrame(columns= [ "PRICE", "QUANTITY"])
        test_best["PRICE"] = np.arange(minn-20, maxx, 0.01)
        test_best["QUANTITY"] = loaded_model.predict(pd.DataFrame(test_best['PRICE']))
        
        # Profit
        # 
        test_best['PROFIT'] = (test_best["PRICE"] - mean_price) * test_best['QUANTITY'] 
        
        
        ind_best = np.where(test_best['PROFIT'] == test_best['PROFIT'].max())[0][0]
        indata_best = test_best.loc[[ind_best]]
        
        
        price_best = np.round([i for i in indata_best['PRICE']][0],2)
        profit_best = np.round([i for i in indata_best['PROFIT']][0],2)
        quantity_best = np.round([i for i in indata_best['QUANTITY']][0],2)






        st.metric(label="Current price", value=pr, delta_color="off")
        st.metric(label="Optimum price", value=price_best, delta_color="off")
    with col2:
        
        
    
        st.metric(label="Expected demand on next week", value=[i for i in current_price['QUANTITY']][0],
        delta_color="off")
        st.metric(label="Optimum quantity", value=quantity_best,
        delta_color="off")
        
    with col3:
        st.metric(label="Next week profit based on current price", value=[i for i in current_price['PROFIT']][0],
        delta_color="off")
        st.metric(label="Next week profit based on optimum price", value=profit_best,
        delta_color="off")
        



    
    
    #############################

    
    

    A, B = st.columns(2)
    
    with A:
        
        st.write('**The trend below shows the change of Price over time**')
        
        TOOLS = 'save,pan,box_zoom,reset,wheel_zoom,hover'
        pq = figure(x_axis_type="datetime", title="Price over time", y_axis_type="linear", plot_height = 350,
                tools = TOOLS, plot_width = 600)
        pq.xaxis.axis_label = 'Time'
        pq.yaxis.axis_label = 'Price per item'
        # p.circle(2010, product_.regular_price.min(), size = 10, color = 'red')

        pq.line(product_.date_week, product_.regular_price,line_color="purple", line_width = 3)
        pq.select_one(HoverTool).tooltips = [
            ('Time', '@x'),
            ('Price per item', '@y'),
        ]

        output_file("reqular_price1.html", title="Line Chart")
        save(pq)
        p1 = open("reqular_price1.html", errors="ignore")
        components.html(p1.read(), height=500, width=700)
        
        
        
        
        # logged_model = 'runs:/a4e3a4228f674dcb865f949651c47bc2/models'

        # Load model as a PyFuncModel.
        # loaded_model = mlflow.pyfunc.load_model(logged_model)

    
        test = pd.DataFrame(columns= [ "PRICE", "QUANTITY"])
        
        # price in step of 0.01 and denominations increase in that sense
        test["PRICE"] = np.arange(minn-20, maxx, 0.01)
        test["QUANTITY"] = loaded_model.predict(pd.DataFrame(test['PRICE']))
        
        # Profit
        # 
        test['PROFIT'] = (test["PRICE"] - mean_price) * test['QUANTITY'] 
        
        
        ind = np.where(test['PROFIT'] == test['PROFIT'].max())[0][0]
        indata = test.loc[[ind]]
        
        
        price = [i for i in indata['PRICE']][0]
        profit = [i for i in indata['PROFIT']][0]
        
        
        
        
        st.write('**The graph shows exact price for maximum profit**')
        
        TOOLS = 'save,pan,box_zoom,reset,wheel_zoom,hover'
        pq = figure(title="Profit over Price", y_axis_type="linear", plot_height = 350,
                tools = TOOLS, plot_width = 600)
        pq.xaxis.axis_label = 'Price'
        pq.yaxis.axis_label = 'Profit'
        # p.circle(2010, product_.regular_price.min(), size = 10, color = 'red')
        # figure out the index point for max profit
        

        pq.line(test.PRICE, test.PROFIT,line_color="purple", line_width = 3)
        pq.select_one(HoverTool).tooltips = [
            ('Price', '@x'),
            ('Profit', '@y'),
        ]
        
     
        

        
        vline = Span(location=price, dimension='height', line_color='red', line_width=1)
        # Horizontal line
        hline = Span(location=profit, dimension='width', line_color='green', line_width=1)

        pq.renderers.extend([vline, hline])
        
        
        
    

        output_file("optimum_price.html", title="Line Chart")
        save(pq)
        p1 = open("optimum_price.html", errors="ignore")
        components.html(p1.read(), height=500, width=700)
        
       
        
        
    
    
    with B:
        
        st.write('**The trend below shows the change of quantity over time**')
        
        TOOLS = 'save,pan,box_zoom,reset,wheel_zoom,hover'
        p = figure(x_axis_type="datetime", title="Quantity over time", y_axis_type="linear", plot_height = 350,
                tools = TOOLS, plot_width = 600)
        p.xaxis.axis_label = 'Time'
        p.yaxis.axis_label = 'Quantity sold'
        # p.circle(2010, product_.regular_price.min(), size = 10, color = 'red')

        p.line(product_.date_week, product_.regular_volume,line_color="purple", line_width = 3)
        p.select_one(HoverTool).tooltips = [
            ('Time', '@x'),
            ('Number of item sold', '@y'),
        ]

        output_file("regular_volume1.html", title="Line Chart")
        save(p)   

        p2 = open("regular_volume1.html", errors="ignore")
        components.html(p2.read(), height=500, width=700)
 
 
 ##########################################
        st.write('**The trend below shows that product is very sensitive to price changes**')
 
        TOOLS = 'save,pan,box_zoom,reset,wheel_zoom,hover'
        paa = figure(title="Quantity over time", y_axis_type="linear", plot_height = 350,
                tools = TOOLS, plot_width = 600)
        paa.xaxis.axis_label = 'Price'
        paa.yaxis.axis_label = 'Quantity'
        # p.circle(2010, product_.regular_price.min(), size = 10, color = 'red')
        # logged_model = 'runs:/a4e3a4228f674dcb865f949651c47bc2/models'

        
        slope = loaded_model.coef_[0]
        intercept = loaded_model.intercept_

        # Make the regression line
        regression_line = Slope(gradient=slope, y_intercept=intercept, line_color="red")
        
        paa.add_layout(regression_line)
        

        paa.scatter(product_.regular_price, product_.regular_volume,line_color="purple", line_width = 3)
        paa.select_one(HoverTool).tooltips = [
            ('Time', '@x'),
            ('Number of item sold', '@y'),
        ]

        paa.add_layout(regression_line)

        output_file("quantity_price.html", title="Line Chart")
        save(paa)   

        paa = open("quantity_price.html", errors="ignore")
        components.html(paa.read(), height=500, width=700)






if option == 'Product_ID: 12062063':
    st.title('Product_ID: 12062063')
    
    loaded_model_prom  = joblib.load('Modelss/model_12062063.pkl') 
    loaded_model = joblib.load('Modelss/model_promo_12062063.pkl') 
    
    Price_Elasticity_regular = np.round(loaded_model.coef_[0], 2)
    Price_Elasticity_promo = np.round(loaded_model_prom.coef_[0], 2)
    
    st.subheader(f'Price Elasticity for regular price: {Price_Elasticity_regular}')
    st.subheader(f'Price Elasticity for promo price: {Price_Elasticity_promo}')
    
    
    df = pd.read_csv('Senior Data Assignment.csv')
    
    columns = [i for i in df.columns]
    sku_cde = [i for i in df['sku_cde'].unique()]
    df_regular = df[columns[0:5]]
    df_regular = df_regular.dropna()
    
    
    
    # product_ = pd.read_csv('Senior Data Assignment.csv')
    product_ = df_regular[df_regular['sku_cde']==111708109]
    product_['date_week'] = pd.to_datetime(product_['date_week'])
    # product_['year'] = pd.to_datetime(product_['date_week']).dt.year
    
    # st.dataframe(product_)
    ##pr---- price
    
    maxx = [i for i in product_[['regular_price']].max()][0]
    minn = [i for i in product_[['regular_price']].min()][0]
    
    ### avarage cost price for the product
    mean_price = np.round(product_['cost_price'].mean(), 2)

    
    mid = (maxx + minn)/2
    
    with st.sidebar:
        
        pr = st.sidebar.slider('Choose price', minn, maxx+1, mid)
        st.write('For Product ID: 111708109, we see that max price was 186 when the optimal price could be 126.47 for maximum profit . this means the Product was loosing in profit. but if they had sold for 126 the sales may decrease, quantity been sold will decrease off course but profit will increase.')
    
        st.write('**Please note that all the products can be interpreted in an similar manner**')
    
        st.write('Now that we have these new price points for each items, once they are set up is important to continuously monitoring the sales and profits made out of these items.')


        st.write('**Elasticities interpretation**')
        
        st.write('A 10% price decrease in Product ID: 111708109 , it increases sales demand by 172.8% or a 10% price increase in Product ID: 62875832, it decreases sales demand by 172.8%')




    col1, col2, col3 = st.columns(3)
    # col1.metric("Temperature", "70 °F", "1.2 °F")
    # col2.metric("Wind", "9 mph", "-8%")
    # col3.metric("Humidity", "86%", "4%")
    
    #############################################################################################
    with col1:
        
        
        
      #####$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        # loaded_model = mlflow.pyfunc.load_model(logged_model)
        
        current_price = pd.DataFrame(columns= [ "PRICE", "QUANTITY"])
        current_price['PRICE'] = [pr]
        current_price['QUANTITY'] = np.round(loaded_model.predict(pd.DataFrame([pr])), 2)
        current_price['PROFIT'] = np.round((current_price["PRICE"] - mean_price) * current_price['QUANTITY'], 2)
        
        
        test = pd.DataFrame(columns= [ "PRICE", "QUANTITY"])
        
        # price in step of 0.01 and denominations increase in that sense
        test["PRICE"] = np.arange(minn, maxx, 0.01)
        test["QUANTITY"] = loaded_model.predict(pd.DataFrame(test['PRICE']))
        
        # Profit
        test['PROFIT'] = (test["PRICE"] - mean_price) * test['QUANTITY'] 
        
        ind = np.where(test['PROFIT'] == test['PROFIT'].max())[0][0]
        indata = test.loc[[ind]]
        
        
        pricee = np.round([i for i in indata['PRICE']][0],2)
        profitt = np.round([i for i in indata['PROFIT']][0], 2)
        quantity = np.round([i for i in indata['QUANTITY']][0], 2)
       
#########################################################################################



        test_best = pd.DataFrame(columns= [ "PRICE", "QUANTITY"])
        test_best["PRICE"] = np.arange(minn-20, maxx, 0.01)
        test_best["QUANTITY"] = loaded_model.predict(pd.DataFrame(test_best['PRICE']))
        
        # Profit
        # 
        test_best['PROFIT'] = (test_best["PRICE"] - mean_price) * test_best['QUANTITY'] 
        
        
        ind_best = np.where(test_best['PROFIT'] == test_best['PROFIT'].max())[0][0]
        indata_best = test_best.loc[[ind_best]]
        
        
        price_best = np.round([i for i in indata_best['PRICE']][0],2)
        profit_best = np.round([i for i in indata_best['PROFIT']][0],2)
        quantity_best = np.round([i for i in indata_best['QUANTITY']][0],2)






        st.metric(label="Current price", value=pr, delta_color="off")
        st.metric(label="Optimum price", value=price_best, delta_color="off")
    with col2:
        
        
    
        st.metric(label="Expected demand on next week", value=[i for i in current_price['QUANTITY']][0],
        delta_color="off")
        st.metric(label="Optimum quantity", value=quantity_best,
        delta_color="off")
        
    with col3:
        st.metric(label="Next week profit based on current price", value=[i for i in current_price['PROFIT']][0],
        delta_color="off")
        st.metric(label="Next week profit based on optimum price", value=profit_best,
        delta_color="off")
        



    
    
    #############################

    
    

    A, B = st.columns(2)
    
    with A:
        
        st.write('**The trend below shows the change of Price over time**')
        
        TOOLS = 'save,pan,box_zoom,reset,wheel_zoom,hover'
        pq = figure(x_axis_type="datetime", title="Price over time", y_axis_type="linear", plot_height = 350,
                tools = TOOLS, plot_width = 600)
        pq.xaxis.axis_label = 'Time'
        pq.yaxis.axis_label = 'Price per item'
        # p.circle(2010, product_.regular_price.min(), size = 10, color = 'red')

        pq.line(product_.date_week, product_.regular_price,line_color="purple", line_width = 3)
        pq.select_one(HoverTool).tooltips = [
            ('Time', '@x'),
            ('Price per item', '@y'),
        ]

        output_file("reqular_price1.html", title="Line Chart")
        save(pq)
        p1 = open("reqular_price1.html", errors="ignore")
        components.html(p1.read(), height=500, width=700)
        
        
        
        
        # logged_model = 'runs:/a4e3a4228f674dcb865f949651c47bc2/models'

        # Load model as a PyFuncModel.
        # loaded_model = mlflow.pyfunc.load_model(logged_model)

    
        test = pd.DataFrame(columns= [ "PRICE", "QUANTITY"])
        
        # price in step of 0.01 and denominations increase in that sense
        test["PRICE"] = np.arange(minn-20, maxx, 0.01)
        test["QUANTITY"] = loaded_model.predict(pd.DataFrame(test['PRICE']))
        
        # Profit
        # 
        test['PROFIT'] = (test["PRICE"] - mean_price) * test['QUANTITY'] 
        
        
        ind = np.where(test['PROFIT'] == test['PROFIT'].max())[0][0]
        indata = test.loc[[ind]]
        
        
        price = [i for i in indata['PRICE']][0]
        profit = [i for i in indata['PROFIT']][0]
        
        
        
        
        st.write('**The graph shows exact price for maximum profit**')
        
        TOOLS = 'save,pan,box_zoom,reset,wheel_zoom,hover'
        pq = figure(title="Profit over Price", y_axis_type="linear", plot_height = 350,
                tools = TOOLS, plot_width = 600)
        pq.xaxis.axis_label = 'Price'
        pq.yaxis.axis_label = 'Profit'
        # p.circle(2010, product_.regular_price.min(), size = 10, color = 'red')
        # figure out the index point for max profit
        

        pq.line(test.PRICE, test.PROFIT,line_color="purple", line_width = 3)
        pq.select_one(HoverTool).tooltips = [
            ('Price', '@x'),
            ('Profit', '@y'),
        ]
        
     
        

        
        vline = Span(location=price, dimension='height', line_color='red', line_width=1)
        # Horizontal line
        hline = Span(location=profit, dimension='width', line_color='green', line_width=1)

        pq.renderers.extend([vline, hline])
        
        
        
    

        output_file("optimum_price.html", title="Line Chart")
        save(pq)
        p1 = open("optimum_price.html", errors="ignore")
        components.html(p1.read(), height=500, width=700)
        
       
        
        
    
    
    with B:
        
        st.write('**The trend below shows the change of quantity over time**')
        
        TOOLS = 'save,pan,box_zoom,reset,wheel_zoom,hover'
        p = figure(x_axis_type="datetime", title="Quantity over time", y_axis_type="linear", plot_height = 350,
                tools = TOOLS, plot_width = 600)
        p.xaxis.axis_label = 'Time'
        p.yaxis.axis_label = 'Quantity sold'
        # p.circle(2010, product_.regular_price.min(), size = 10, color = 'red')

        p.line(product_.date_week, product_.regular_volume,line_color="purple", line_width = 3)
        p.select_one(HoverTool).tooltips = [
            ('Time', '@x'),
            ('Number of item sold', '@y'),
        ]

        output_file("regular_volume1.html", title="Line Chart")
        save(p)   

        p2 = open("regular_volume1.html", errors="ignore")
        components.html(p2.read(), height=500, width=700)
 
 
 ##########################################
        st.write('**The trend below shows that product is very sensitive to price changes**')
 
        TOOLS = 'save,pan,box_zoom,reset,wheel_zoom,hover'
        paa = figure(title="Quantity over time", y_axis_type="linear", plot_height = 350,
                tools = TOOLS, plot_width = 600)
        paa.xaxis.axis_label = 'Price'
        paa.yaxis.axis_label = 'Quantity'
        # p.circle(2010, product_.regular_price.min(), size = 10, color = 'red')
        # logged_model = 'runs:/a4e3a4228f674dcb865f949651c47bc2/models'

        
        slope = loaded_model.coef_[0]
        intercept = loaded_model.intercept_

        # Make the regression line
        regression_line = Slope(gradient=slope, y_intercept=intercept, line_color="red")
        
        paa.add_layout(regression_line)
        

        paa.scatter(product_.regular_price, product_.regular_volume,line_color="purple", line_width = 3)
        paa.select_one(HoverTool).tooltips = [
            ('Time', '@x'),
            ('Number of item sold', '@y'),
        ]

        paa.add_layout(regression_line)

        output_file("quantity_price.html", title="Line Chart")
        save(paa)   

        paa = open("quantity_price.html", errors="ignore")
        components.html(paa.read(), height=500, width=700)





if option == 'Product_ID: 130680236':
    st.title('Product_ID: 130680236')
    
    loaded_model_prom  = joblib.load('Modelss/model_regular_130680236.pkl') 
    loaded_model = joblib.load('Modelss/model_promo_130680236.pkl') 
    
    Price_Elasticity_regular = np.round(loaded_model.coef_[0], 2)
    Price_Elasticity_promo = np.round(loaded_model_prom.coef_[0], 2)
    
    st.subheader(f'Price Elasticity for regular price: {Price_Elasticity_regular}')
    st.subheader(f'Price Elasticity for promo price: {Price_Elasticity_promo}')
    
    
    df = pd.read_csv('Senior Data Assignment.csv')
    
    columns = [i for i in df.columns]
    sku_cde = [i for i in df['sku_cde'].unique()]
    df_regular = df[columns[0:5]]
    df_regular = df_regular.dropna()
    
    
    
    # product_ = pd.read_csv('Senior Data Assignment.csv')
    product_ = df_regular[df_regular['sku_cde']==130680236]
    product_['date_week'] = pd.to_datetime(product_['date_week'])
    # product_['year'] = pd.to_datetime(product_['date_week']).dt.year
    
    # st.dataframe(product_)
    ##pr---- price
    
    maxx = [i for i in product_[['regular_price']].max()][0]
    minn = [i for i in product_[['regular_price']].min()][0]
    
    ### avarage cost price for the product
    mean_price = np.round(product_['cost_price'].mean(), 2)

    
    mid = (maxx + minn)/2
    
    with st.sidebar:
        
        pr = st.sidebar.slider('Choose price', minn, maxx+1, mid)
        st.write('For Product ID: 111708109, we see that max price was 186 when the optimal price could be 126.47 for maximum profit . this means the Product was loosing in profit. but if they had sold for 126 the sales may decrease, quantity been sold will decrease off course but profit will increase.')
    
        st.write('**Please note that all the products can be interpreted in an similar manner**')
    
        st.write('Now that we have these new price points for each items, once they are set up is important to continuously monitoring the sales and profits made out of these items.')


        st.write('**Elasticities interpretation**')
        
        st.write('A 10% price decrease in Product ID: 111708109 , it increases sales demand by 172.8% or a 10% price increase in Product ID: 62875832, it decreases sales demand by 172.8%')



    col1, col2, col3 = st.columns(3)
    # col1.metric("Temperature", "70 °F", "1.2 °F")
    # col2.metric("Wind", "9 mph", "-8%")
    # col3.metric("Humidity", "86%", "4%")
    
    #############################################################################################
    with col1:
        
        
        
      #####$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        # loaded_model = mlflow.pyfunc.load_model(logged_model)
        
        current_price = pd.DataFrame(columns= [ "PRICE", "QUANTITY"])
        current_price['PRICE'] = [pr]
        current_price['QUANTITY'] = np.round(loaded_model.predict(pd.DataFrame([pr])), 2)
        current_price['PROFIT'] = np.round((current_price["PRICE"] - mean_price) * current_price['QUANTITY'], 2)
        
        
        test = pd.DataFrame(columns= [ "PRICE", "QUANTITY"])
        
        # price in step of 0.01 and denominations increase in that sense
        test["PRICE"] = np.arange(minn, maxx, 0.01)
        test["QUANTITY"] = loaded_model.predict(pd.DataFrame(test['PRICE']))
        
        # Profit
        test['PROFIT'] = (test["PRICE"] - mean_price) * test['QUANTITY'] 
        
        ind = np.where(test['PROFIT'] == test['PROFIT'].max())[0][0]
        indata = test.loc[[ind]]
        
        
        pricee = np.round([i for i in indata['PRICE']][0],2)
        profitt = np.round([i for i in indata['PROFIT']][0], 2)
        quantity = np.round([i for i in indata['QUANTITY']][0], 2)
       
#########################################################################################



        test_best = pd.DataFrame(columns= [ "PRICE", "QUANTITY"])
        test_best["PRICE"] = np.arange(minn-20, maxx, 0.01)
        test_best["QUANTITY"] = loaded_model.predict(pd.DataFrame(test_best['PRICE']))
        
        # Profit
        # 
        test_best['PROFIT'] = (test_best["PRICE"] - mean_price) * test_best['QUANTITY'] 
        
        
        ind_best = np.where(test_best['PROFIT'] == test_best['PROFIT'].max())[0][0]
        indata_best = test_best.loc[[ind_best]]
        
        
        price_best = np.round([i for i in indata_best['PRICE']][0],2)
        profit_best = np.round([i for i in indata_best['PROFIT']][0],2)
        quantity_best = np.round([i for i in indata_best['QUANTITY']][0],2)






        st.metric(label="Current price", value=pr, delta_color="off")
        st.metric(label="Optimum price", value=price_best, delta_color="off")
    with col2:
        
        
    
        st.metric(label="Expected demand on next week", value=[i for i in current_price['QUANTITY']][0],
        delta_color="off")
        st.metric(label="Optimum quantity", value=quantity_best,
        delta_color="off")
        
    with col3:
        st.metric(label="Next week profit based on current price", value=[i for i in current_price['PROFIT']][0],
        delta_color="off")
        st.metric(label="Next week profit based on optimum price", value=profit_best,
        delta_color="off")
        



    
    
    #############################

    
    

    A, B = st.columns(2)
    
    with A:
        
        st.write('**The trend below shows the change of Price over time**')
        
        TOOLS = 'save,pan,box_zoom,reset,wheel_zoom,hover'
        pq = figure(x_axis_type="datetime", title="Price over time", y_axis_type="linear", plot_height = 350,
                tools = TOOLS, plot_width = 600)
        pq.xaxis.axis_label = 'Time'
        pq.yaxis.axis_label = 'Price per item'
        # p.circle(2010, product_.regular_price.min(), size = 10, color = 'red')

        pq.line(product_.date_week, product_.regular_price,line_color="purple", line_width = 3)
        pq.select_one(HoverTool).tooltips = [
            ('Time', '@x'),
            ('Price per item', '@y'),
        ]

        output_file("reqular_price1.html", title="Line Chart")
        save(pq)
        p1 = open("reqular_price1.html", errors="ignore")
        components.html(p1.read(), height=500, width=700)
        
        
        
        
        # logged_model = 'runs:/a4e3a4228f674dcb865f949651c47bc2/models'

        # Load model as a PyFuncModel.
        # loaded_model = mlflow.pyfunc.load_model(logged_model)

    
        test = pd.DataFrame(columns= [ "PRICE", "QUANTITY"])
        
        # price in step of 0.01 and denominations increase in that sense
        test["PRICE"] = np.arange(minn-20, maxx, 0.01)
        test["QUANTITY"] = loaded_model.predict(pd.DataFrame(test['PRICE']))
        
        # Profit
        # 
        test['PROFIT'] = (test["PRICE"] - mean_price) * test['QUANTITY'] 
        
        
        ind = np.where(test['PROFIT'] == test['PROFIT'].max())[0][0]
        indata = test.loc[[ind]]
        
        
        price = [i for i in indata['PRICE']][0]
        profit = [i for i in indata['PROFIT']][0]
        
        
        
        
        st.write('**The graph shows exact price for maximum profit**')
        
        TOOLS = 'save,pan,box_zoom,reset,wheel_zoom,hover'
        pq = figure(title="Profit over Price", y_axis_type="linear", plot_height = 350,
                tools = TOOLS, plot_width = 600)
        pq.xaxis.axis_label = 'Price'
        pq.yaxis.axis_label = 'Profit'
        # p.circle(2010, product_.regular_price.min(), size = 10, color = 'red')
        # figure out the index point for max profit
        

        pq.line(test.PRICE, test.PROFIT,line_color="purple", line_width = 3)
        pq.select_one(HoverTool).tooltips = [
            ('Price', '@x'),
            ('Profit', '@y'),
        ]
        
     
        

        
        vline = Span(location=price, dimension='height', line_color='red', line_width=1)
        # Horizontal line
        hline = Span(location=profit, dimension='width', line_color='green', line_width=1)

        pq.renderers.extend([vline, hline])
        
        
        
    

        output_file("optimum_price.html", title="Line Chart")
        save(pq)
        p1 = open("optimum_price.html", errors="ignore")
        components.html(p1.read(), height=500, width=700)
        
       
        
        
    
    
    with B:
        
        st.write('**The trend below shows the change of quantity over time**')
        
        TOOLS = 'save,pan,box_zoom,reset,wheel_zoom,hover'
        p = figure(x_axis_type="datetime", title="Quantity over time", y_axis_type="linear", plot_height = 350,
                tools = TOOLS, plot_width = 600)
        p.xaxis.axis_label = 'Time'
        p.yaxis.axis_label = 'Quantity sold'
        # p.circle(2010, product_.regular_price.min(), size = 10, color = 'red')

        p.line(product_.date_week, product_.regular_volume,line_color="purple", line_width = 3)
        p.select_one(HoverTool).tooltips = [
            ('Time', '@x'),
            ('Number of item sold', '@y'),
        ]

        output_file("regular_volume1.html", title="Line Chart")
        save(p)   

        p2 = open("regular_volume1.html", errors="ignore")
        components.html(p2.read(), height=500, width=700)
 
 
 ##########################################
        st.write('**The trend below shows that product is very sensitive to price changes**')
 
        TOOLS = 'save,pan,box_zoom,reset,wheel_zoom,hover'
        paa = figure(title="Quantity over time", y_axis_type="linear", plot_height = 350,
                tools = TOOLS, plot_width = 600)
        paa.xaxis.axis_label = 'Price'
        paa.yaxis.axis_label = 'Quantity'
        # p.circle(2010, product_.regular_price.min(), size = 10, color = 'red')
        # logged_model = 'runs:/a4e3a4228f674dcb865f949651c47bc2/models'

        
        slope = loaded_model.coef_[0]
        intercept = loaded_model.intercept_

        # Make the regression line
        regression_line = Slope(gradient=slope, y_intercept=intercept, line_color="red")
        
        paa.add_layout(regression_line)
        

        paa.scatter(product_.regular_price, product_.regular_volume,line_color="purple", line_width = 3)
        paa.select_one(HoverTool).tooltips = [
            ('Time', '@x'),
            ('Number of item sold', '@y'),
        ]

        paa.add_layout(regression_line)

        output_file("quantity_price.html", title="Line Chart")
        save(paa)   

        paa = open("quantity_price.html", errors="ignore")
        components.html(paa.read(), height=500, width=700)






if option == 'Product_ID: 137353695':
    st.title('Product_ID: 137353695')
    
    loaded_model_prom  = joblib.load('Modelss/model_regular_137353695.pkl') 
    loaded_model = joblib.load('Modelss/model_promo_137353695.pkl') 
    
    Price_Elasticity_regular = np.round(loaded_model.coef_[0], 2)
    Price_Elasticity_promo = np.round(loaded_model_prom.coef_[0], 2)
    
    st.subheader(f'Price Elasticity for regular price: {Price_Elasticity_regular}')
    st.subheader(f'Price Elasticity for promo price: {Price_Elasticity_promo}')
    
    
    df = pd.read_csv('Senior Data Assignment.csv')
    
    columns = [i for i in df.columns]
    sku_cde = [i for i in df['sku_cde'].unique()]
    df_regular = df[columns[0:5]]
    df_regular = df_regular.dropna()
    
    
    
    # product_ = pd.read_csv('Senior Data Assignment.csv')
    product_ = df_regular[df_regular['sku_cde']==137353695]
    product_['date_week'] = pd.to_datetime(product_['date_week'])
    # product_['year'] = pd.to_datetime(product_['date_week']).dt.year
    
    # st.dataframe(product_)
    ##pr---- price
    
    maxx = [i for i in product_[['regular_price']].max()][0]
    minn = [i for i in product_[['regular_price']].min()][0]
    
    ### avarage cost price for the product
    mean_price = np.round(product_['cost_price'].mean(), 2)

    
    mid = (maxx + minn)/2
    
    with st.sidebar:
        
        pr = st.sidebar.slider('Choose price', minn, maxx+1, mid)
        st.write('For Product ID: 111708109, we see that max price was 186 when the optimal price could be 126.47 for maximum profit . this means the Product was loosing in profit. but if they had sold for 126 the sales may decrease, quantity been sold will decrease off course but profit will increase.')
    
        st.write('**Please note that all the products can be interpreted in an similar manner**')
    
        st.write('Now that we have these new price points for each items, once they are set up is important to continuously monitoring the sales and profits made out of these items.')


        st.write('**Elasticities interpretation**')
        
        st.write('A 10% price decrease in Product ID: 111708109 , it increases sales demand by 172.8% or a 10% price increase in Product ID: 62875832, it decreases sales demand by 172.8%')


    col1, col2, col3 = st.columns(3)
    # col1.metric("Temperature", "70 °F", "1.2 °F")
    # col2.metric("Wind", "9 mph", "-8%")
    # col3.metric("Humidity", "86%", "4%")
    
    #############################################################################################
    with col1:
        
        
        
      #####$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        # loaded_model = mlflow.pyfunc.load_model(logged_model)
        
        current_price = pd.DataFrame(columns= [ "PRICE", "QUANTITY"])
        current_price['PRICE'] = [pr]
        current_price['QUANTITY'] = np.round(loaded_model.predict(pd.DataFrame([pr])), 2)
        current_price['PROFIT'] = np.round((current_price["PRICE"] - mean_price) * current_price['QUANTITY'], 2)
        
        
        test = pd.DataFrame(columns= [ "PRICE", "QUANTITY"])
        
        # price in step of 0.01 and denominations increase in that sense
        test["PRICE"] = np.arange(minn, maxx, 0.01)
        test["QUANTITY"] = loaded_model.predict(pd.DataFrame(test['PRICE']))
        
        # Profit
        test['PROFIT'] = (test["PRICE"] - mean_price) * test['QUANTITY'] 
        
        ind = np.where(test['PROFIT'] == test['PROFIT'].max())[0][0]
        indata = test.loc[[ind]]
        
        
        pricee = np.round([i for i in indata['PRICE']][0],2)
        profitt = np.round([i for i in indata['PROFIT']][0], 2)
        quantity = np.round([i for i in indata['QUANTITY']][0], 2)
       
#########################################################################################



        test_best = pd.DataFrame(columns= [ "PRICE", "QUANTITY"])
        test_best["PRICE"] = np.arange(minn-20, maxx, 0.01)
        test_best["QUANTITY"] = loaded_model.predict(pd.DataFrame(test_best['PRICE']))
        
        # Profit
        # 
        test_best['PROFIT'] = (test_best["PRICE"] - mean_price) * test_best['QUANTITY'] 
        
        
        ind_best = np.where(test_best['PROFIT'] == test_best['PROFIT'].max())[0][0]
        indata_best = test_best.loc[[ind_best]]
        
        
        price_best = np.round([i for i in indata_best['PRICE']][0],2)
        profit_best = np.round([i for i in indata_best['PROFIT']][0],2)
        quantity_best = np.round([i for i in indata_best['QUANTITY']][0],2)






        st.metric(label="Current price", value=pr, delta_color="off")
        st.metric(label="Optimum price", value=price_best, delta_color="off")
    with col2:
        
        
    
        st.metric(label="Expected demand on next week", value=[i for i in current_price['QUANTITY']][0],
        delta_color="off")
        st.metric(label="Optimum quantity", value=quantity_best,
        delta_color="off")
        
    with col3:
        st.metric(label="Next week profit based on current price", value=[i for i in current_price['PROFIT']][0],
        delta_color="off")
        st.metric(label="Next week profit based on optimum price", value=profit_best,
        delta_color="off")
        



    
    
    #############################

    
    

    A, B = st.columns(2)
    
    with A:
        
        st.write('**The trend below shows the change of Price over time**')
        
        TOOLS = 'save,pan,box_zoom,reset,wheel_zoom,hover'
        pq = figure(x_axis_type="datetime", title="Price over time", y_axis_type="linear", plot_height = 350,
                tools = TOOLS, plot_width = 600)
        pq.xaxis.axis_label = 'Time'
        pq.yaxis.axis_label = 'Price per item'
        # p.circle(2010, product_.regular_price.min(), size = 10, color = 'red')

        pq.line(product_.date_week, product_.regular_price,line_color="purple", line_width = 3)
        pq.select_one(HoverTool).tooltips = [
            ('Time', '@x'),
            ('Price per item', '@y'),
        ]

        output_file("reqular_price1.html", title="Line Chart")
        save(pq)
        p1 = open("reqular_price1.html", errors="ignore")
        components.html(p1.read(), height=500, width=700)
        
        
        
        
        # logged_model = 'runs:/a4e3a4228f674dcb865f949651c47bc2/models'

        # Load model as a PyFuncModel.
        # loaded_model = mlflow.pyfunc.load_model(logged_model)

    
        test = pd.DataFrame(columns= [ "PRICE", "QUANTITY"])
        
        # price in step of 0.01 and denominations increase in that sense
        test["PRICE"] = np.arange(minn-20, maxx, 0.01)
        test["QUANTITY"] = loaded_model.predict(pd.DataFrame(test['PRICE']))
        
        # Profit
        # 
        test['PROFIT'] = (test["PRICE"] - mean_price) * test['QUANTITY'] 
        
        
        ind = np.where(test['PROFIT'] == test['PROFIT'].max())[0][0]
        indata = test.loc[[ind]]
        
        
        price = [i for i in indata['PRICE']][0]
        profit = [i for i in indata['PROFIT']][0]
        
        
        
        
        st.write('**The graph shows exact price for maximum profit**')
        
        TOOLS = 'save,pan,box_zoom,reset,wheel_zoom,hover'
        pq = figure(title="Profit over Price", y_axis_type="linear", plot_height = 350,
                tools = TOOLS, plot_width = 600)
        pq.xaxis.axis_label = 'Price'
        pq.yaxis.axis_label = 'Profit'
        # p.circle(2010, product_.regular_price.min(), size = 10, color = 'red')
        # figure out the index point for max profit
        

        pq.line(test.PRICE, test.PROFIT,line_color="purple", line_width = 3)
        pq.select_one(HoverTool).tooltips = [
            ('Price', '@x'),
            ('Profit', '@y'),
        ]
        
     
        

        
        vline = Span(location=price, dimension='height', line_color='red', line_width=1)
        # Horizontal line
        hline = Span(location=profit, dimension='width', line_color='green', line_width=1)

        pq.renderers.extend([vline, hline])
        
        
        
    

        output_file("optimum_price.html", title="Line Chart")
        save(pq)
        p1 = open("optimum_price.html", errors="ignore")
        components.html(p1.read(), height=500, width=700)
        
       
        
        
    
    
    with B:
        
        st.write('**The trend below shows the change of quantity over time**')
        
        TOOLS = 'save,pan,box_zoom,reset,wheel_zoom,hover'
        p = figure(x_axis_type="datetime", title="Quantity over time", y_axis_type="linear", plot_height = 350,
                tools = TOOLS, plot_width = 600)
        p.xaxis.axis_label = 'Time'
        p.yaxis.axis_label = 'Quantity sold'
        # p.circle(2010, product_.regular_price.min(), size = 10, color = 'red')

        p.line(product_.date_week, product_.regular_volume,line_color="purple", line_width = 3)
        p.select_one(HoverTool).tooltips = [
            ('Time', '@x'),
            ('Number of item sold', '@y'),
        ]

        output_file("regular_volume1.html", title="Line Chart")
        save(p)   

        p2 = open("regular_volume1.html", errors="ignore")
        components.html(p2.read(), height=500, width=700)
 
 
 ##########################################
        st.write('**The trend below shows that product is very sensitive to price changes**')
 
        TOOLS = 'save,pan,box_zoom,reset,wheel_zoom,hover'
        paa = figure(title="Quantity over time", y_axis_type="linear", plot_height = 350,
                tools = TOOLS, plot_width = 600)
        paa.xaxis.axis_label = 'Price'
        paa.yaxis.axis_label = 'Quantity'
        # p.circle(2010, product_.regular_price.min(), size = 10, color = 'red')
        # logged_model = 'runs:/a4e3a4228f674dcb865f949651c47bc2/models'

        
        slope = loaded_model.coef_[0]
        intercept = loaded_model.intercept_

        # Make the regression line
        regression_line = Slope(gradient=slope, y_intercept=intercept, line_color="red")
        
        paa.add_layout(regression_line)
        

        paa.scatter(product_.regular_price, product_.regular_volume,line_color="purple", line_width = 3)
        paa.select_one(HoverTool).tooltips = [
            ('Time', '@x'),
            ('Number of item sold', '@y'),
        ]

        paa.add_layout(regression_line)

        output_file("quantity_price.html", title="Line Chart")
        save(paa)   

        paa = open("quantity_price.html", errors="ignore")
        components.html(paa.read(), height=500, width=700)




