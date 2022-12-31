
# ------------------------------------------------------------------------------------------------
# Libraries
# ------------------------------------------------------------------------------------------------

import pandas as pd
import geopandas
import streamlit as st
import numpy as np
import folium
from PIL import Image
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster
import plotly.express as px
from datetime import datetime

# ------------------------------------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------------------------------------


st.set_page_config(
page_title = 'Home',
page_icon = '🏠',
layout = 'wide')


image = Image.open('logo.png')
st.sidebar.image(image, width = 180)

st.sidebar.markdown('# House Rocket')
st.sidebar.markdown('## Realizando o seu sonho da casa própria')
st.sidebar.markdown("""---""")



# ------------------------------------------------------------------------------------------------
# Function
# ------------------------------------------------------------------------------------------------

@st.cache(allow_output_mutation = True)

def get_data(path):
    data = pd.read_csv(path)
    
    return data

@st.cache(allow_output_mutation = True)

def get_geofile(url):
    geofile = geopandas.read_file(url)

    return geofile

def set_feature(data):
    # Add new Features
    data['price_m2'] = data['price'] / data['sqft_lot']

    return data

def overview_data(data):


    # ------------------------------------------------------------------------------------------------
    # Definição de Filtros
    # ------------------------------------------------------------------------------------------------

    f_attributes = st.sidebar.multiselect('Enter columns', data.columns)
    f_zipcode = st.sidebar.multiselect('Enter zipcode', data['zipcode'].unique())

    st.markdown('### Overview Data and Analysis')

    tab1, tab2, tab3 = st.tabs(["📈 Data Overview", "📈 Average Values", "💻 Descriptive Analysis"])

    with tab1:
        

        if (f_zipcode != []) & (f_attributes != []):
            data = data.loc[data['zipcode'].isin(f_zipcode) , f_attributes]

        elif (f_zipcode != []) & (f_attributes == []):
            data = data.loc[data['zipcode'].isin(f_zipcode), :]

        elif (f_zipcode == []) & (f_attributes != []):
            data = data.loc[:, f_attributes]

        else:
            data = data.copy()
        
        st.header('Overview Data')
        st.dataframe(data)

    with tab2:
    

        # ------------------------------------------------------------------------------------------------
        # Average Metrics
        # ------------------------------------------------------------------------------------------------

        df1 = data[['id', 'zipcode']].groupby('zipcode').count().reset_index()
        df2 = data[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
        df3 = data[['sqft_living', 'zipcode']].groupby('zipcode').mean().reset_index()
        df4 = data[['price_m2', 'zipcode']].groupby('zipcode').mean().reset_index()

        # Merge

        m1 = pd.merge(df1, df2, on = 'zipcode', how = 'inner')
        m2 = pd.merge(m1, df3, on = 'zipcode', how = 'inner')
        df = pd.merge(m2, df4, on = 'zipcode', how = 'inner')

        # Alterar o nome das colunas
        df.columns = ['ZIPCODE', 'TOTAL HOUSES', 'PRICE' , 'SQRT LIVING', 'PRICE / M2']

        st.header('Average Values')
        st.dataframe(df, height = 600)


    with tab3:

        # ------------------------------------------------------------------------------------------------
        # Statistic Descriptive
        # ------------------------------------------------------------------------------------------------

        num_attributes = data.select_dtypes( include = ['int64', 'float64'])
        media = pd.DataFrame(num_attributes.apply(np.mean))
        mediana = pd.DataFrame(num_attributes.apply(np.median))
        desvio_padrao = pd.DataFrame(num_attributes.apply(np.std))
        maximo = pd.DataFrame(num_attributes.apply(np.max))
        minimo = pd.DataFrame(num_attributes.apply(np.min))

        df1 = pd.concat([media, mediana, desvio_padrao, maximo, minimo], axis=1).reset_index()
        df1.columns = ['attributes','mean', 'median', 'std', 'max', 'min']

        st.header('Descriptive Analysis')
        st.dataframe(df1, height = 600)


    return None


def portfolio_density(data, geofile):


    st.markdown("""___""")
    st.markdown('### Portfolio')

    tab1, tab2 = st.tabs(['🌎 Portfolio Density','💵 Price Density'])

    with tab1:

        df = data.sample(50)

        density_map = folium.Map(location = [data['lat'].mean(),
                                    data['long'].mean()],
                                    default_zoom_start = 15)

        marker_cluster = MarkerCluster().add_to( density_map )
        for name, row in df.iterrows():
            folium.Marker([row['lat'], row['long']],
                    popup = 'Sold R${0} on: {1}. Features:{2} sqft, {3} bedrooms, {4} bathrooms, year built: {5}'.format( row['price'],
                                                                                                                row['date'],
                                                                                                                row['sqft_living'],
                                                                                                                row['bedrooms'],
                                                                                                                row['bathrooms'],
                                                                                                                row['yr_built'])).add_to( marker_cluster )
                


        folium_static(density_map)


    with tab2:

        df = data[[ 'price', 'zipcode']].groupby('zipcode').mean().reset_index()
        df.columns = ['ZIP', 'PRICE']

        df = df.sample(10)

        geofile = geofile[geofile['ZIP'].isin(df['ZIP'].tolist())]

        region_price_map = folium.Map( location = [data['lat'].mean(), data['long'].mean()], default_zoom_start = 15)

        region_price_map.choropleth(data = df,
                                    geo_data = geofile,
                                    columns = ['ZIP', 'PRICE'],
                                    key_on = 'feature.properties.ZIP',
                                    fill_color = 'YlOrRd',
                                    fill_opacity = 0.7,
                                    line_opacity = 0.2,
                                    legend_name = 'AVG PRICE')
        
        folium_static(region_price_map)

    return None

def commercial_distribution(data):

    # Average Price per Year Built

    # Setup Filters
    min_year_built = int(data['yr_built'].min())
    max_year_built = int(data['yr_built'].max())

    # Mensagem
    st.sidebar.subheader('Select Max Year Built')

    # Definir Filtro
    f_year_built = st.sidebar.slider('Year Built', min_year_built,
                                                    max_year_built,
                                                    min_year_built)

    # Get Date
    data['date'] = pd.to_datetime(data['date']).dt.strftime('%Y-%m-%d')

    # Definir valores mínimos e máximos para filtro de data
    min_date = datetime.strptime( data['date'].min(), '%Y-%m-%d' )
    max_date = datetime.strptime( data['date'].max(), '%Y-%m-%d' )

    # Mensagem
    st.sidebar.subheader('Select Max Date Built')

    # Definir Filtro
    f_date = st.sidebar.slider('Date', min_date, max_date, max_date)

    # Histograma

    # Definir valores mínimos, máximos e média para filtro de preço
    min_price = int(data['price'].min())
    max_price = int(data['price'].max())
    avg_price = int(data['price'].mean())

    # Mensagem
    st.sidebar.subheader('Select Max Price')

    # Definir Filtro
    f_price = st.sidebar.slider('Price', min_price, max_price, avg_price)

    # ------------------------------------------------------------------------------------------------
    # Distribuição dos imóveis por categorias comerciais
    # ------------------------------------------------------------------------------------------------

    st.sidebar.title('Commercial Option')

    st.markdown("""___""")
    st.markdown('### Commercial Atribuites')

    # Converter formato data 
    data['date'] = pd.to_datetime(data['date']).dt.strftime('%Y-%-m-%d')

    # ------------------------------------------------------------------------------------------------
    # Average Price per Year
    # ------------------------------------------------------------------------------------------------

    # Filrar dataframe conforme aplicação do filtro

    tab1, tab2, tab3 = st.tabs(['🗓 Average Price per Year Built', '📅 Average Price per Day', '💵 Price Distribution'])

    with tab1:
        df = data.loc[data['yr_built'] < f_year_built]

        # Agrupar dados para plotar o gráfico
        df = data[['yr_built', 'price']].groupby('yr_built').mean().reset_index()

        # Plotar gráfico
        fig = px.line(df, x = 'yr_built', y = 'price')

        st.plotly_chart(fig, use_container_widthe = True)


    # ------------------------------------------------------------------------------------------------
    # Average Price per Day
    # ------------------------------------------------------------------------------------------------

    with tab2:
        # Converter Date para formato Datetime
        data['date'] = pd.to_datetime(data['date'])

        # Filrar dataframe conforme aplicação do filtro
        df = data.loc[data['date'] < f_date]

        # Agrupar dados para plotaro gráfico
        df = df[['date', 'price']].groupby('date').mean().reset_index()

        # Plotar gráfico
        fig = px.line(df, x= 'date', y= 'price')

        st.plotly_chart(fig, use_container_widthe = True)


    # ------------------------------------------------------------------------------------------------
    # Histograma Preço
    # ------------------------------------------------------------------------------------------------

    # Filrar dataframe conforme aplicação do filtro
    
    with tab3:
        df = data.loc[data['price'] > f_price]

        # Plotar gráfico
        fig = px.histogram(df, x='price', nbins=50)

        st.header('Price Distribution')
        st.plotly_chart(fig, use_container_widthe = True)

    return None

def attributes_distribution(data):
    

    # Categorias Físicas

    # Mensagem
    st.sidebar.subheader('Attributes Options')

    # Filter Bedrooms
    f_bedrooms = st.sidebar.selectbox('Max number of bedrooms', 
                                        sorted(set(data['bedrooms'].unique())))

    # Filter Bathrooms
    f_bathrooms = st.sidebar.selectbox('Max number of bathrooms',
                                        sorted(set(data['bathrooms'].unique())))


    # Filter Floors
    f_floors = st.sidebar.selectbox('Max number of floors',
                                        sorted(set(data['floors'].unique())))


    # House per water view
    f_water_view = st.sidebar.checkbox('Only Houses with Water Vier')


    # ------------------------------------------------------------------------------------------------
    # Distribuição dos imóveis por categorias físicas
    # ------------------------------------------------------------------------------------------------

    st.markdown("""___""")
    st.markdown('### House Atributes')

    tab1, tab2, tab3, tab4 = st.tabs(['🛏 Bedrooms', '🛁 Bathrooms', '🏢 Floors', '🏖 Water Front'])

    with tab1:
        # Bedrooms

        # Filtrar dataframe conforme aplicação do filtro
        df = data[data['bedrooms'] > f_bedrooms]

        # Plotar gráfico
        fig = px.histogram(df, x = 'bedrooms', nbins = 19)
        st.markdown('#### Houses per Bedrooms')
        st.plotly_chart(fig, use_container_width = True)

    with tab2:
        # Bathrooms
        
        # Filtrar dataframe conforme aplicação do filtro
        df = data[data['bathrooms'] > f_bathrooms]   

        # Plotar gráfico
        fig = px.histogram(data, x = 'bathrooms', nbins = 19)
        st.markdown('#### Houses per Bathrooms')
        st.plotly_chart(fig, use_container_width = True)


    with tab3:
        # Floors

        # Filtrar dataframe conforme aplicação do filtro
        df = data[data['floors'] > f_floors]

        # Plotar gráfico
        fig = px.histogram(df, x = 'floors', nbins = 19)

        st.markdown('#### Houses per Floors')
        st.plotly_chart(fig, use_container_width = True)

    with tab4:
        # Water View

        # Filtrar dataframe conforme aplicação do filtro
        if f_water_view:
            df = data[data['waterfront'] == 1]
        else:
            df = data.copy()


        # Plotar gráfico
        fig = px.histogram(df, x = 'waterfront', nbins = 10)

        st.markdown('#### Houses per Water Front')
        st.plotly_chart(fig, use_container_width = True)

    return None


if __name__ == '__main__':
    # ETL
    
    # Data Extration
    path = 'datasets/kc_house_data.csv'
    url = 'https://opendata.arcgis.com/datasets/83fc2e72903343aabff6de8cb445b81c_2.geojson'

    # Load Data
    data = get_data(path)
    geofile = get_geofile(url)

    # Transformation
    data = set_feature(data)

    overview_data(data)

    portfolio_density(data, geofile)

    commercial_distribution(data)

    attributes_distribution(data)
