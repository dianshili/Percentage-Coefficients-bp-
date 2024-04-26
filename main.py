import pandas as pd
import numpy as np
import torch


class pure_regression_bp_comparation():

    def __init__(self, data: pd.DataFrame, data_columns: list, n_bootstrap: int, alpha=0.05):
        self.path_collect = []
        self.coef_container = torch.tensor([])
        self.n_bootstrap = n_bootstrap
        self.data_pd = data
        self.data_torch = torch.tensor(self.data_pd[data_columns].values)
        self.data_columns = {v: counter for counter, v in enumerate(data_columns)}
        self.alpha = alpha
        self.clusters = None

    def _ols(self, A, B):
        # OLS regression using the updated lstsq method
        result = torch.linalg.lstsq(B, A)
        coefficients = result.solution[:, 0]
        predictions = B @ coefficients
        return coefficients, predictions

    def _p2star(self,p):
        if p < 0.001:
            return '***'
        elif p < 0.01:
            return '**'
        elif p < 0.05:
            return '*'
        else:
            return ''

    def bp_Norm(self, df):
        return (df - df.min()) / (df.max() - df.min())
    def regression(self, Y: str, X_list: list):
        self.X = self.data_torch[:, [list(self.data_columns.keys()).index(item) for item in X_list if item in X_list]]
        self.n = len(self.X)
        self.Y = self.data_torch[:, self.data_columns[Y]].unsqueeze(1)
        self.X_intercept = torch.cat((torch.ones_like(self.Y), self.X), dim=1)
        self.coef_container_ = torch.zeros((self.n_bootstrap, self.X_intercept.size(-1) - 1))
        self.indices = torch.randint(0, self.n, (self.n, self.n_bootstrap))
        X_sample = self.X_intercept[self.indices]
        Y_sample = self.Y[self.indices]
        for i in range(self.n_bootstrap):
            X_sample_ = X_sample[:, i]
            Y_sample_ = Y_sample[:, i]
            coeff, _ = self._ols(Y_sample_, X_sample_)
            self.coef_container_[i] = coeff[1:]
        self.coef_container = torch.cat((self.coef_container, self.coef_container_), dim=1)
        self.path_collect.extend(['{}->{}'.format(i, Y) for count, i in enumerate(X_list)])

    def update(self):
        source = pd.Series([i[0] for i in list(map(lambda x: x.split('->'), self.path_collect))])
        target = pd.Series([i[-1] for i in list(map(lambda x: x.split('->'), self.path_collect))])
        intersection = list(set(source) & set(target))
        for i in intersection:
            target_intern = target[target == i].index
            source_intern = source[source == i].index
            self.path_collect.extend(['{}->{}'.format(
                self.path_collect[k],
                list(map(lambda x: x.split('->'), self.path_collect))[j][-1])
                for k in target_intern for j in source_intern])
            self.coef_container_ = torch.cat((
                [(self.coef_container[:, k] * self.coef_container[:, j]).unsqueeze(-1)
                 for k in target_intern for j in source_intern]
            ), dim=1)
            self.coef_container = torch.cat(
                (self.coef_container, self.coef_container_), dim=1)


    def get_statistics(self):
        statistics = {}
        for counter, i in enumerate(self.path_collect):
            statistics.update(
                {i: {
                    'value': self.coef_container[:, counter].mean().cpu().item(),
                    '95lower': torch.quantile(self.coef_container[:, counter], 0.05 / 2).cpu().item(),
                    '95upper': torch.quantile(self.coef_container[:, counter], 1 - 0.05 / 2).cpu().item(),
                    'std': self.coef_container[:, counter].std().cpu().item(),
                    'max': self.coef_container[:, counter].max().cpu().item(),
                    'min': self.coef_container[:, counter].min().cpu().item(),
                    'significant_p': 2 * min((self.coef_container[:, counter] <= 0).sum() / self.n_bootstrap,
                                             (self.coef_container[:,
                                              counter] >= 0).sum() / self.n_bootstrap).cpu().item()
                }}
            )
        self.statistics = statistics

    def scalar_comparation(self):
        scalar_result = {}
        for counter_i, i in enumerate(self.path_collect):
            for counter_j, j in enumerate(self.path_collect):
                tem_tensor = self.coef_container[:, counter_i].abs() - self.coef_container[:, counter_j].abs()
                scalar_result.update(
                    {(i, j):
                         {'value': tem_tensor.mean().cpu().item(),
                          '95lower': torch.quantile(tem_tensor, 0.05 / 2).cpu().item(),
                          '95upper': torch.quantile(tem_tensor, 1 - 0.05 / 2).cpu().item(),
                          'std': tem_tensor.std().cpu().item(),
                          'max': tem_tensor.max().cpu().item(),
                          'min': tem_tensor.min().cpu().item(),
                          'significant_p': 2 * min((tem_tensor <= 0).sum() / self.n_bootstrap,
                                                   (tem_tensor >= 0).sum() / self.n_bootstrap).cpu().item()}
                     }
                )
        self.scalar_result = scalar_result

    def directional_comparation(self):
        directional_result = {}
        for counter_i, i in enumerate(self.path_collect):
            for counter_j, j in enumerate(self.path_collect):
                tem_tensor = self.coef_container[:, counter_i] - self.coef_container[:, counter_j]
                directional_result.update(
                    {(i, j):
                         {'value': tem_tensor.mean().cpu().item(),
                          '95lower': torch.quantile(tem_tensor, 0.05 / 2).cpu().item(),
                          '95upper': torch.quantile(tem_tensor, 1 - 0.05 / 2).cpu().item(),
                          'std': tem_tensor.std().cpu().item(),
                          'max': tem_tensor.max().cpu().item(),
                          'min': tem_tensor.min().cpu().item(),
                          'significant_p': 2 * min((tem_tensor <= 0).sum() / self.n_bootstrap,
                                                   (tem_tensor >= 0).sum() / self.n_bootstrap).cpu().item()}
                     }
                )
        self.directional_result = directional_result

    def forward(self):
        self.get_statistics()
        self.scalar_comparation()
        self.directional_comparation()
        scalar_result_pd_dict, directional_result_pd_dict = {}, {}
        coef_result_pd_dict = {}
        for counter_i, i in enumerate(self.path_collect):
            scalar_result_pd_dict.update(
                {i:
                     {j:
                          str(np.around(self.scalar_result[(i, j)]['value'], 3)) + \
                          self._p2star(self.scalar_result[(i, j)]['significant_p'])
                      for counter_j, j in enumerate(self.path_collect)}})
            directional_result_pd_dict.update(
                {i:
                     {j:
                          str(np.around(self.directional_result[(i, j)]['value'], 3)) + \
                          self._p2star(self.directional_result[(i, j)]['significant_p'])
                      for counter_j, j in enumerate(self.path_collect)}})
            coef_result_pd_dict.update(
                {i: {
                    'coeff': str(np.around(self.statistics[i]['value'], 3)) + self._p2star(
                        self.statistics[i]['significant_p']),
                    'std': self.statistics[i]['std'],
                    '95%CI': '[{},{}]'.format(self.statistics[i]['95lower'], self.statistics[i]['95upper'])
                }}
            )
        self.scalar_result_format = pd.DataFrame(scalar_result_pd_dict)
        self.directional_result_format = pd.DataFrame(directional_result_pd_dict)
        self.coef_result_format = pd.DataFrame(coef_result_pd_dict).T

    def format(self, path='./'):
        self.scalar_result_format.to_excel(path + 'scalar_comparation_result_format.xlsx')
        self.directional_result_format.to_excel(path + 'directional_comparation_result_format.xlsx')
        self.coef_result_format.to_excel(path + 'regression_result_format.xlsx')


if __name__ == '__main__':
    data = pd.read_excel('depression.xlsx')
    columns = ['PT_01',
               'BirthGender',
               'Age_01',
               'Edu2',
               'Edu3',
               'Edu4',
               'marital',
               'Fincome',
               'PCC_01',
               'OPPC_01',
               'PatientA_01',
               'WB_01']

    data = data[columns]
    data_torch = torch.from_numpy(data.values)
    data_columns = {c: counter for counter, c in enumerate(columns)}
    f = pure_regression_bp_comparation(data, columns, n_bootstrap=5000)
    f.regression(Y='PT_01', X_list=['BirthGender',
                                    'Age_01',
                                    'Edu2',
                                    'Edu3',
                                    'Edu4',
                                    'marital',
                                    'Fincome',
                                    'PCC_01',
                                    'OPPC_01'])
    f.regression(Y='PatientA_01', X_list=['BirthGender',
                                          'Age_01',
                                          'Edu2',
                                          'Edu3',
                                          'Edu4',
                                          'marital',
                                          'Fincome',
                                          'PCC_01',
                                          'OPPC_01',
                                          'PT_01'])
    f.update()
    f.regression(Y='WB_01', X_list=['BirthGender',
                                    'Age_01',
                                    'Edu2',
                                    'Edu3',
                                    'Edu4',
                                    'marital',
                                    'Fincome',
                                    'PCC_01',
                                    'OPPC_01',
                                    'PT_01',
                                    'PatientA_01'])
    f.update()
    f.forward()
    f.format(path='./')
