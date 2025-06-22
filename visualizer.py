import plotly.express as px

class Visualizer:

    @staticmethod
    def scatter(df, hover_data_key="word", color_key="animal"):
        common_settings = {
            'data_frame': df,
            'x': 'x',
            'y': 'y',
            'color': color_key,
            'hover_data': [hover_data_key],
            'title': 'Embedding visualization'
        }

        if 'z' in df.columns:
            fig = px.scatter_3d(z='z', **common_settings)
        else:
            fig = px.scatter(**common_settings)

        fig.update_traces(textposition='top center', marker=dict(size=10, opacity=0.9))
        fig.update_layout(template='seaborn', hoverlabel=dict(font=dict(size=20)))
        fig.show()
