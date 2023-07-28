import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import skew
from scipy.optimize import minimize
import plotly.graph_objs as go
import multiprocessing

app = dash.Dash(__name__)

def get_portfolio_returns(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    returns = data.pct_change().dropna()
    return returns

def get_portfolio_skewness(returns):
    return skew(returns, axis=0, nan_policy='omit')

def portfolio_return(weights, returns):
    return np.sum(returns.mean() * weights) * 252

def portfolio_volatility(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)

def negative_portfolio_sharpe(weights, returns, cov_matrix, risk_free_rate):
    p_return = portfolio_return(weights, returns)
    p_volatility = portfolio_volatility(weights, cov_matrix)
    return -(p_return - risk_free_rate) / p_volatility

def simulate_portfolios_async(num_portfolios, returns, cov_matrix, risk_free_rate, return_dict):
    results, _ = simulate_portfolios(num_portfolios, returns, cov_matrix, risk_free_rate)
    return_dict['results'] = results

def simulate_portfolios(num_portfolios, returns, cov_matrix, risk_free_rate):
    results = np.zeros((4, num_portfolios))
    weights_record = np.zeros((len(returns.columns), num_portfolios))
    for i in range(num_portfolios):
        weights = np.random.random(len(returns.columns))
        weights /= np.sum(weights)
        weights_record[:, i] = weights

        portfolio_return_i = portfolio_return(weights, returns)
        portfolio_volatility_i = portfolio_volatility(weights, cov_matrix)
        portfolio_sharpe_i = (portfolio_return_i - risk_free_rate) / portfolio_volatility_i

        results[0, i] = portfolio_return_i
        results[1, i] = portfolio_volatility_i
        results[2, i] = portfolio_sharpe_i
        results[3, i] = get_portfolio_skewness(returns @ weights)

    return results, weights_record

# Defina os ativos (tickers) que você deseja obter os retornos
tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']

# Defina as datas de início e fim
start_date = '2023-01-01'
end_date = '2023-07-01'

# Obtenha os retornos da carteira de investimentos
portfolio_returns = get_portfolio_returns(tickers, start_date, end_date)

# Calcule a matriz de covariância dos retornos
cov_matrix = portfolio_returns.cov()

# Calcule a assimetria da carteira
portfolio_skewness = get_portfolio_skewness(portfolio_returns)

# Taxa livre de risco (utilizada para cálculo do índice de Sharpe)
risk_free_rate = 0.03

# Defina os limites para os pesos dos ativos na carteira (entre 0 e 1)
bounds = tuple((0, 1) for _ in range(len(tickers)))

# Restrição para garantir que a soma dos pesos seja igual a 1
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

# Encontre os pesos ótimos para maximizar o índice de Sharpe
initial_weights = np.ones(len(tickers)) / len(tickers)
result = minimize(negative_portfolio_sharpe, initial_weights, args=(portfolio_returns, cov_matrix, risk_free_rate),
                  method='SLSQP', bounds=bounds, constraints=constraints)

optimal_weights = result.x
optimal_return = portfolio_return(optimal_weights, portfolio_returns)
optimal_volatility = portfolio_volatility(optimal_weights, cov_matrix)

app.layout = html.Div(children=[
    html.H1(children='Retornos dos Ativos da Carteira'),

    dcc.Dropdown(
        id='ticker-dropdown',
        options=[{'label': ticker, 'value': ticker} for ticker in ['Todos'] + tickers],
        value='Todos',
        multi=False
    ),

    dcc.Graph(id='returns-graph'),

    dcc.Graph(
        id='skewness-graph',
        figure={
            'data': [
                {'x': portfolio_returns.index, 'y': portfolio_skewness, 'type': 'scatter', 'mode': 'lines', 'name': 'Assimetria'}
            ],
            'layout': {
                'title': 'Assimetria da Carteira',
                'xaxis': {'title': 'Data'},
                'yaxis': {'title': 'Assimetria'},
                'hovermode': 'closest'
            }
        }
    ),

    dcc.Graph(
        id='risk-return-graph',
        figure={
            'data': [
                go.Scatter(
                    x=portfolio_returns.mean() * 252,
                    y=portfolio_returns.std() * np.sqrt(252),
                    mode='markers',
                    marker=dict(size=12),
                    text=portfolio_returns.columns
                ),
                go.Scatter(
                    x=[optimal_return],
                    y=[optimal_volatility],
                    mode='markers',
                    marker=dict(size=16, color='red'),
                    text='Carteira Ótima'
                )
            ],
            'layout': {
                'title': 'Risco e Retorno da Carteira',
                'xaxis': {'title': 'Retorno Esperado'},
                'yaxis': {'title': 'Desvio Padrão (Risco)'},
                'hovermode': 'closest'
            }
        }
    ),

    dcc.Graph(id='simulation-graph')
])

@app.callback(
    Output('returns-graph', 'figure'),
    [Input('ticker-dropdown', 'value')]
)
def update_graph(selected_ticker):
    if selected_ticker == 'Todos':
        data_to_plot = portfolio_returns
    else:
        data_to_plot = pd.DataFrame(portfolio_returns[selected_ticker])

    figure = {
        'data': [
            {'x': data_to_plot.index, 'y': data_to_plot[ticker], 'name': ticker}
            for ticker in data_to_plot.columns
        ],
        'layout': {
            'title': 'Retornos Diários dos Ativos',
            'xaxis': {'title': 'Data'},
            'yaxis': {'title': 'Retorno'},
            'hovermode': 'closest'
        }
    }
    return figure

@app.callback(
    Output('simulation-graph', 'figure'),
    [Input('simulation-graph', 'id')],
    [State('simulation-graph', 'figure')]
)
def update_simulation_graph(id, existing_figure):
    if id is not None:
        num_portfolios = 1000  # Você pode ajustar o número de carteiras simuladas aqui

        # Utilize o multiprocessing para realizar a simulação de carteiras em segundo plano
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        process = multiprocessing.Process(target=simulate_portfolios_async, args=(num_portfolios, portfolio_returns, cov_matrix, risk_free_rate, return_dict))
        process.start()
        process.join()

        # Obtenha os resultados da simulação
        results = return_dict['results']
        simulated_returns, simulated_volatility, simulated_sharpe, simulated_skewness = results

        trace = go.Scatter(
            x=simulated_volatility,
            y=simulated_returns,
            mode='markers',
            marker=dict(size=8, color=simulated_sharpe, colorbar=dict(title='Índice de Sharpe')),
            text=simulated_skewness,
            hoverinfo='text'
        )

        if existing_figure:
            existing_figure['data'] = [trace]
            return existing_figure
        else:
            return {
                'data': [trace],
                'layout': {
                    'title': 'Simulação de Carteiras - Risco e Retorno',
                    'xaxis': {'title': 'Desvio Padrão (Risco)'},
                    'yaxis': {'title': 'Retorno Esperado'},
                    'hovermode': 'closest'
                }
            }

if __name__ == '__main__':
    app.run_server(debug=True)