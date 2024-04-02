import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.dates import AutoDateLocator, DateFormatter, HOURLY
import numpy as np

class Metrics:

    def __init__(self, df, title, energy_ylim, free_energy_ylim, metrics_ylim, error_ylim, eps_ylim, hour):
        df['obstime'] = df['obstime'].apply(pd.to_datetime)
        self.df = df
        self.title = title
        self.energy_ylim = energy_ylim
        self.free_energy_ylim = free_energy_ylim
        self.metrics_ylim = metrics_ylim
        self.error_ylim = error_ylim
        self.eps_ylim = eps_ylim
        self.hour = hour
        pred_E = self.df['pred_E_1e33']
        ref_E = self.df['ref_E_1e33']
        self.eps = np.array(pred_E)/np.array(ref_E)


    def print(self):
        df = self.df
        eps_mean = self.eps.mean()
        c_vec_mean = df['C_vec'].mean()
        c_cs_mean = df['C_cs'].mean()
        e_n_mean = df["E_n_prime"].mean()
        e_m_mean = df["E_m_prime"].mean()
        l2_err_mean = df['l2_err'].mean()

        print()
        print(f"avg C_vec     : {c_vec_mean:.2f}")
        print(f"avg C_cs      : {c_cs_mean:.2f}")
        print(f"avg E_n_prime : {e_n_mean:.2f}")
        print(f"avg E_m_prime : {e_m_mean:.2f}")
        print(f"avg eps       : {eps_mean:.2f}")
        print(f"avg l2_err    : {l2_err_mean:.2f}")
        print()
    

    def plot(self, result_path):
        self.draw_energy(result_path)
        self.draw_free_energy(result_path)
        self.draw_metrics(result_path)
        self.draw_error(result_path)
        self.draw_eps(result_path)
        self.draw_bnorm(result_path)


    def draw_energy(self, result_path):
        df = self.df
        title = self.title
        ylim = self.energy_ylim

        obstime = df['obstime']
        pred_E = df['pred_E_1e33']
        ref_E = df['ref_E_1e33']

        fig, ax = plt.subplots(figsize=(12, 6))

        marker_style = dict(linestyle='-', markersize=3, fillstyle='full')
        text_style = dict(fontsize=16, fontdict={'family': 'monospace'})
        ax.tick_params(labelsize=14)
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))

        # ascribe the data to the axes
        ax.plot(obstime, pred_E,'o', **marker_style, color='red', label='pred')
        ax.plot(obstime, ref_E,'o', **marker_style, color='blue', label='ISEE')

        # format the x-axis with universal time
        locator = AutoDateLocator()
        locator.intervald[HOURLY] = [self.hour] # only show every 3 hours
        formatter = DateFormatter('%H')
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        ax.set_xlabel('time (hour)',**text_style)
        ax.set_ylabel('energy (erg) / 1e33 ',**text_style)
        ax.set_title(f'{title} / mag energy starting at {str(obstime[0])}',**text_style)
        ax.set_ylim(ylim)
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig(result_path / 'energy.png', dpi=600)

    def draw_free_energy(self, result_path):
        df = self.df
        title = self.title
        ylim = self.free_energy_ylim

        obstime = df['obstime']
        try : 
            pot_E = df['pot_E_1e33']
        except:
            return
        pred_free_E = df['pred_E_1e33'] - pot_E
        ref_free_E = df['ref_E_1e33'] - pot_E

        fig, ax = plt.subplots(figsize=(12, 6))

        marker_style = dict(linestyle='-', markersize=3, fillstyle='full')
        text_style = dict(fontsize=16, fontdict={'family': 'monospace'})
        ax.tick_params(labelsize=14)
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))

        # ascribe the data to the axes
        ax.plot(obstime, pred_free_E,'o', **marker_style, color='red', label='pred')
        ax.plot(obstime, ref_free_E,'o', **marker_style, color='blue', label='ISEE')

        # format the x-axis with universal time
        locator = AutoDateLocator()
        locator.intervald[HOURLY] = [self.hour] # only show every 3 hours
        formatter = DateFormatter('%H')
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        ax.set_xlabel('time (hour)',**text_style)
        ax.set_ylabel('free energy (erg) / 1e33 ',**text_style)
        ax.set_title(f'{title} / mag free energy starting at {str(obstime[0])}',**text_style)
        ax.set_ylim(ylim)
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig(result_path / 'free_energy.png', dpi=600)


    def draw_metrics(self, result_path):
        df = self.df
        title = self.title
        ylim = self.metrics_ylim

        obstime = df['obstime']
        C_vec = df['C_vec']
        C_cs = df['C_cs']
        E_n = df["E_n_prime"]
        E_m = df["E_m_prime"]

        fig, ax = plt.subplots(figsize=(12, 6))

        marker_style = dict(linestyle='-', markersize=3, fillstyle='full')
        text_style = dict(fontsize=16, fontdict={'family': 'monospace'})
        ax.tick_params(labelsize=14)
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))

        # ascribe the data to the axes
        ax.plot(obstime, C_vec,'o', **marker_style, color='red', label='C_vec')
        ax.plot(obstime, C_cs,'o', **marker_style, color='blue', label='C_cs')
        ax.plot(obstime, E_n,'o', **marker_style, color='green', label="E_n'")
        ax.plot(obstime, E_m,'o', **marker_style, color='orange', label="E_m'")

        # format the x-axis with universal time
        locator = AutoDateLocator()
        locator.intervald[HOURLY] = [self.hour] # only show every 3 hours
        formatter = DateFormatter('%H')
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        ax.set_xlabel('time (hour)',**text_style)
        ax.set_ylabel('metric',**text_style)
        ax.set_title(f'{title} / metrics starting at {str(obstime[0])}',**text_style)

        ax.set_ylim(ylim)
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig(result_path / 'metrics.png', dpi=600)


    def draw_error(self, result_path):
        df = self.df
        title = self.title
        ylim = self.error_ylim

        obstime = df['obstime']
        l2_err = df['l2_err']

        fig, ax = plt.subplots(figsize=(12, 6))

        marker_style = dict(linestyle='-', markersize=3, fillstyle='full')
        text_style = dict(fontsize=16, fontdict={'family': 'monospace'})
        ax.tick_params(labelsize=14)
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))

        # ascribe the data to the axes
        ax.plot(obstime, l2_err,'o', **marker_style, color='red', label='rel_err')

        # format the x-axis with universal time
        locator = AutoDateLocator()
        locator.intervald[HOURLY] = [self.hour] # only show every 3 hours
        formatter = DateFormatter('%H')
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        ax.set_xlabel('time (hour)',**text_style)
        ax.set_ylabel('relative error',**text_style)
        ax.set_title(f'{title} / relative error starting at {str(obstime[0])}',**text_style)

        ax.set_ylim(ylim)
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig(result_path / 'error.png', dpi=600)
    

    def draw_eps(self, result_path):
        df = self.df
        title = self.title
        ylim = self.eps_ylim

        obstime = df['obstime']

        fig, ax = plt.subplots(figsize=(12, 6))

        marker_style = dict(linestyle='-', markersize=3, fillstyle='full')
        text_style = dict(fontsize=16, fontdict={'family': 'monospace'})
        ax.tick_params(labelsize=14)
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))

        # ascribe the data to the axes
        ax.plot(obstime, self.eps,'o', **marker_style, color='red', label='eps')

        # format the x-axis with universal time
        locator = AutoDateLocator()
        locator.intervald[HOURLY] = [self.hour] # only show every 3 hours
        formatter = DateFormatter('%H')
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        ax.set_xlabel('time (hour)',**text_style)
        ax.set_ylabel('eps ',**text_style)
        ax.set_title(f'{title} / total mag energy starting at {str(obstime.iloc[0])}',**text_style)
        ax.set_ylim(ylim)
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig(result_path / 'eps.png', dpi=600)

    def draw_bnorm(self, result_path):
        df = self.df
        title = self.title
        # ylim = self.eps_ylim

        obstime = df['obstime']

        fig, ax = plt.subplots(figsize=(12, 6))

        marker_style = dict(linestyle='-', markersize=3, fillstyle='full')
        text_style = dict(fontsize=16, fontdict={'family': 'monospace'})
        ax.tick_params(labelsize=14)
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))

        # ascribe the data to the axes
        ax.plot(obstime, df['b_norm'],'o', **marker_style, color='red', label='eps')

        # format the x-axis with universal time
        locator = AutoDateLocator()
        locator.intervald[HOURLY] = [self.hour] # only show every 3 hours
        formatter = DateFormatter('%H')
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        ax.set_xlabel('time (hour)',**text_style)
        ax.set_ylabel(' b_norm (G) ',**text_style)
        ax.set_title(f'{title} / b_norm starting at {str(obstime.iloc[0])}',**text_style)
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig(result_path / 'b_norm.png', dpi=600)



class MetricsPaper:

    def __init__(self, df, ref_df, title, energy_all_ylim, energy_ylim, eps_ylim, hour):
        df['obstime'] = df['obstime'].apply(pd.to_datetime)
        ref_df['obstime'] = ref_df['obstime'].apply(pd.to_datetime)
        self.df = df
        self.ref_df = ref_df
        self.title = title
        self.energy_all_ylim = energy_all_ylim
        self.energy_ylim = energy_ylim
        self.eps_ylim = eps_ylim
        self.hour = hour
        self.merged_df = pd.merge(df, ref_df, on='obstime', how='inner')
        pred_E = self.merged_df['pred_E_1e33_x']
        ref_E = self.merged_df['ref_E_1e33']
        self.eps = np.array(pred_E)/np.array(ref_E)


    def print(self):
        eps = self.eps
        eps_mean = eps.mean()

        print()
        print(f"avg eps    : {eps_mean:.2f}")
        print()


    def plot(self, result_path):
        self.draw_all_energy(result_path)
        self.draw_energy(result_path)
        self.draw_eps(result_path)

    
    def draw_all_energy(self, result_path):
        df = self.df
        ref_df = self.ref_df
        title = self.title
        ylim = self.energy_all_ylim


        obstime = df['obstime']
        pred_E = df['pred_E_1e33']

        ref_obstime = ref_df['obstime']
        ref_E = ref_df['ref_E_1e33']

        fig, ax = plt.subplots(figsize=(12, 6))

        marker_style = dict(linestyle='-', markersize=3, fillstyle='full')
        text_style = dict(fontsize=16, fontdict={'family': 'monospace'})
        ax.tick_params(labelsize=14)
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))

        # ascribe the data to the axes
        ax.plot(obstime, pred_E,'o', **marker_style, color='red', label='pred')
        ax.plot(ref_obstime, ref_E,'o', **marker_style, color='blue', label='ISEE')

        # format the x-axis with universal time
        locator = AutoDateLocator()
        locator.intervald[HOURLY] = [self.hour] # only show every 3 hours
        formatter = DateFormatter('%H')
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        ax.set_xlabel('time (hour)',**text_style)
        ax.set_ylabel('energy (erg) / 1e33 ',**text_style)
        ax.set_title(f'{title} / total mag energy starting at {str(obstime.iloc[0])}',**text_style)
        ax.set_ylim(ylim)
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig(result_path / 'all_energy.png', dpi=600)


    def draw_energy(self, result_path):
        df = self.df
        ref_df = self.ref_df
        title = self.title
        ylim = self.energy_ylim

        check = (df['obstime'] >= ref_df['obstime'].iloc[0]) & (df['obstime'] <= ref_df['obstime'].iloc[-1])

        obstime = df['obstime'][check]
        pred_E = df['pred_E_1e33'][check]

        ref_obstime = ref_df['obstime']
        ref_E = ref_df['ref_E_1e33']

        fig, ax = plt.subplots(figsize=(12, 6))

        marker_style = dict(linestyle='-', markersize=3, fillstyle='full')
        text_style = dict(fontsize=16, fontdict={'family': 'monospace'})
        ax.tick_params(labelsize=14)
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))

        # ascribe the data to the axes
        ax.plot(obstime, pred_E,'o', **marker_style, color='red', label='pred')
        ax.plot(ref_obstime, ref_E,'o', **marker_style, color='blue', label='ISEE')

        # format the x-axis with universal time
        locator = AutoDateLocator()
        locator.intervald[HOURLY] = [self.hour] # only show every 3 hours
        formatter = DateFormatter('%H')
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        ax.set_xlabel('time (hour)',**text_style)
        ax.set_ylabel('energy (erg) / 1e33 ',**text_style)
        ax.set_title(f'{title} / total mag energy starting at {str(obstime.iloc[0])}',**text_style)
        ax.set_ylim(ylim)
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig(result_path / 'energy.png', dpi=600)


    def draw_eps(self, result_path):
        merged_df = self.merged_df
        title = self.title
        ylim = self.eps_ylim

        obstime = merged_df['obstime']

        fig, ax = plt.subplots(figsize=(12, 6))

        marker_style = dict(linestyle='-', markersize=3, fillstyle='full')
        text_style = dict(fontsize=16, fontdict={'family': 'monospace'})
        ax.tick_params(labelsize=14)
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))

        # ascribe the data to the axes
        ax.plot(obstime, self.eps,'o', **marker_style, color='red', label='eps')

        # format the x-axis with universal time
        locator = AutoDateLocator()
        locator.intervald[HOURLY] = [self.hour] # only show every 3 hours
        formatter = DateFormatter('%H')
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        ax.set_xlabel('time (hour)',**text_style)
        ax.set_ylabel('energy (erg) / 1e33 ',**text_style)
        ax.set_title(f'{title} / total mag energy starting at {str(obstime.iloc[0])}',**text_style)
        ax.set_ylim(ylim)
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig(result_path / 'eps.png', dpi=600)