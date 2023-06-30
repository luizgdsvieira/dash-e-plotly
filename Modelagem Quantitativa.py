import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import numpy as np

# Dados simulados para exemplo
returns = np.array([0.1, 0.15, 0.12, 0.08, 0.11])
volatility = np.array([0.2, 0.3, 0.25, 0.18, 0.22])
assets = ['Ativo 1', 'Ativo 2', 'Ativo 3', 'Ativo 4', 'Ativo 5']

# Calcular a Fronteira Eficiente
weights = []
port_returns = []
port_volatility = []
num_assets = len(returns)
num_portfolios = 5000

for _ in range(num_portfolios):
    weights.append(np.random.random(num_assets))
    weights[-1] /= np.sum(weights[-1])
    port_returns.append(np.dot(weights[-1], returns))
    port_volatility.append(np.sqrt(np.dot(weights[-1].T, np.dot(np.cov(returns), weights[-1]))))

# Criar o aplicativo Dash
app = dash.Dash(__name__)

# Layout do aplicativo Dash
app.layout = html.Div([
    html.H1('Fronteira Eficiente'),
    html.H4('''
            Olá me chamo Luiz Gustavo da Silva Vieira. Esse é um mini projeto de Carteira
            de investimentos, com o objetivo de apresentar sua Fronteira Eficiente. 
            É apenas um rascunho para estudo pessoal do uso dos pacotes Plotly e Dash do Python. 
            '''),
    
    dcc.Graph(
        id='efficient-frontier',
        figure={
            'data': [
                go.Scatter(
                    x=port_volatility,
                    y=port_returns,
                    mode='markers',
                    marker=dict(size=5),
                    text=[f'Ativo {i+1}' for i in range(num_assets)],
                    hovertemplate='<b>%{text}</b><br>Volatilidade: %{x}<br>Retorno: %{y}<extra></extra>'
                )
            ],
            'layout': go.Layout(
                title='Fronteira Eficiente',
                xaxis={'title': 'Volatilidade'},
                yaxis={'title': 'Retorno'},
                hovermode='closest'
            )
        }
    )
# Logo abaixo segue outros gráficos para análise de carteira

])

# Executar o aplicativo Dash
if __name__ == '__main__':
    app.run_server(debug=True)
