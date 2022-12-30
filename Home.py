import streamlit as st
from PIL import Image


st.set_page_config(
    page_title = 'Home',
    page_icon = '🏠',
    layout = 'wide')


image = Image.open('logo.png')
st.sidebar.image(image, width = 180)

st.sidebar.markdown('# House Rocket')
st.sidebar.markdown('## Realizando o seu sonho da casa própria')
st.sidebar.markdown("""---""")

st.write('# House Rocket - Portfolio Dashboard')
st.markdown(
    """
    Portfolio Dashboard foi construído para acompanhar as principais métricas e caracteristicas de um portfólio de imóveis.
    - Localização;
        - Zip code;
        - Dados geográficos (latitude e logitude).
    - Preço;
    - Ano de construção;
    - Ano de renovação;
    - Atributos dos imóveis:
        - Número de quartos;
        - Número de banheiros;
        - Número de andares;
        - Área Construída;
        - Vista para água;
        
    ### Ask for Help
    - Time de Data Science no Discord
    @flsantos
    """)