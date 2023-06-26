from dash import Dash, html, dcc, Input, Output
import plotly.express as px
import pandas as pd

app = Dash(__name__)

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options

df = pd.read_excel("Vendas.xlsx")


# criando o gráfico
fig = px.bar(df, x="Produto", y="Quantidade", color="ID Loja", barmode="group")
opcoes = list(df['ID Loja'].unique())
opcoes.append("todas as lojas")

app.layout = html.Div(children=[
    html.H1(children='Faturamento das lojas'),
    html.H2(children='Gráfico com o faturamento de todos os produtos separados por loja'),
    html.Div(children='''
        Obs.: Esse gráfico mostra a quantidade de produtos vendidos, não faturamento.
    '''),

    dcc.Dropdown(opcoes, value='todas as lojas', id='listalojas'),

    dcc.Graph(
        id='grafico_quantidade_vendas',
        figure=fig
    )
])

@app.callback(
    Output('grafico_quantidade_vendas', 'figure'),
    Input('listalojas', 'value')
)
def update_output(value):
    if value == "todas as lojas":
        fig = px.bar(df, x="Produto", y="Quantidade", color="ID Loja", barmode="group")
    else: 
        tabela_filtrada = df.loc[df['ID Loja'] == value,  :]    
        fig = px.bar(tabela_filtrada, x="Produto", y="Quantidade", color="ID Loja", barmode="group")    
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)