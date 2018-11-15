import pandas as pd
import numpy as np
from datetime import date
from pandas.tseries.offsets import *
from MorningStar.ExtractMorningStar import ExtractMorningStar
from MySqlConnect import MySqlConnect
from WebData import WebData


class LoadMorningStar(object):
    def __init__(self):
        self._mstr = ExtractMorningStar()
        self._wd = WebData()
        self._ms = MySqlConnect()

    def datacheck_emptyset(self, portfolio):
        portfolio = portfolio
        datacheck = portfolio.copy()
        datacheck.ItemValue = pd.to_numeric(
            datacheck.ItemValue, errors='coerce')
        datacheck = datacheck.groupby(
            ['Ticker', 'Category']).ItemValue.sum().reset_index()
        datacheck = datacheck[(datacheck.ItemValue == 0)]

        if datacheck.shape[0] != 0:
            empty_df = pd.merge(
                portfolio,
                datacheck[['Ticker', 'Category', 'ItemValue']].rename(
                    columns={'ItemValue': 'EmptyFlag'}),
                'left', ['Ticker', 'Category']
            )
            portfolio = empty_df[empty_df.EmptyFlag.isnull()][[
                'EffectiveDate', 'Category', 'Ticker', 'Item', 'ItemValue',
                'AssetType']]

        return portfolio

    def datacheck_incomplete(self, portfolio):
        portfolio = portfolio
        test_category = [
            'Asset Allocation', 'Market Capitalization', 'Equity Sectors',
            'Credit Quality', 'Bond Sectors', 'Country Breakdown', 'Maturity',
            'Coupons', 'Market Classification', 'Super Region', 'World Region'
        ]

        datacheck = portfolio.copy()
        datacheck.ItemValue = pd.to_numeric(
            datacheck.ItemValue, errors='coerce')
        datacheck = datacheck[datacheck.Category.isin(test_category)].groupby(
            ['Ticker', 'Category']).ItemValue.sum().reset_index()
        datacheck = datacheck[
            (datacheck.ItemValue.round(2) < 100) & (datacheck.ItemValue != 0)]

        if datacheck.shape[0] != 0:
            datacheck = pd.merge(datacheck, portfolio[
                ['EffectiveDate', 'Category', 'Ticker',
                 'AssetType']].drop_duplicates(), 'inner',
                                 ['Ticker', 'Category'])
            datacheck['ItemValue'] = 100.00 - datacheck.ItemValue
            datacheck['ItemValue'] = datacheck['ItemValue'].round(2)
            datacheck['Item'] = 'Unknown'
            return datacheck
        else:
            return None

    def load_morningstar_fund_attributes(self, end_date, tickers=None):
        mstr = self._mstr
        ms = self._ms
        end_date = pd.to_datetime(end_date)
        start_date = date.strftime(
            MonthBegin().rollback(end_date - DateOffset(months=1)),
            '%Y-%m-%d'
        )
        end_date = date.strftime(end_date, '%Y-%m-%d')

        if tickers is None:
            tickers_df = ms.mysql_getdata(
                """
                select distinct h.Ticker
                from Portfolio.holdings h
                join Portfolio.securitymaster s
                    on s.Ticker  = h.Ticker
                where HoldingDate between '{}' and '{}'
                    and isCash = 0
                    and h.Ticker is not null
                    and s.SecurityType != 'Cash'
                    and s.SecurityType != 'Equity'
                """.format(start_date, end_date)
            )

            tickers = list(tickers_df.Ticker)

        pd.DataFrame(columns=['EffectiveDate', 'Ticker', 'Table']).to_csv(
            "C:/Users/1/Documents/MySqlDB/LoadFiles/datacheck_empty.csv",
            index=False)

        morningstar = []
        for idx, tk in enumerate(tickers):
            print "{} / {}, {}".format(idx + 1, len(tickers), tk)
            fund_name = mstr.remove_non_ascii(mstr._fund_name(tk))
            update_date = mstr._update_date(tk)

            asset_allc = mstr._asset_allocation(tk)
            bd_style = mstr._bond_style(tk)
            bd_sector = mstr._bond_sector_weightings(tk)
            bd_country = mstr._bond_country_holding(tk)
            bd_maturity = mstr._bond_maturity(tk)
            bd_coupon = mstr._bond_coupon(tk)
            eq_style = mstr._equity_style(tk)
            eq_sector = mstr._equity_sector_weightings(tk)
            eq_regions = mstr._equity_world_regions(tk)

            sec_char = []
            for tb in [asset_allc, bd_style, bd_sector, bd_country, bd_maturity,
                       bd_coupon, eq_style, eq_sector, eq_regions]:
                if tb is not None:
                    sec_char.append(tb)

            sec_char.append(
                pd.DataFrame(
                    {'Portfolio': fund_name,
                     'Table': 'Security Description'},
                    index=['Name']
                ))

            sec_char = pd.concat(sec_char).reset_index()

            sec_char = sec_char.rename(columns={'index': 'Item'})

            sec_char.Item = sec_char.Item.str.replace(
                "%|\*", "").str.strip()

            sec_char.Table = sec_char.Table.str.replace(
                " &", ""
            )

            sec_char = sec_char.replace(u'\u2014', np.nan)
            sec_char['Ticker'] = tk
            sec_char['EffectiveDate'] = pd.to_datetime(update_date)

            sec_char = sec_char[sec_char.Portfolio.notnull()]

            morningstar.append(sec_char)

        morningstar = pd.concat(morningstar)

        morningstar.EffectiveDate = morningstar.EffectiveDate.dt.strftime(
            "%Y-%m-%d")
        morningstar = morningstar.rename(columns={
            'Table': 'Category', 'Portfolio': 'ItemValue'
        })

        incomplete = self.datacheck_incomplete(morningstar)
        morningstar = self.datacheck_emptyset(morningstar)

        morningstar = pd.concat([morningstar, incomplete])
        morningstar = morningstar.reset_index(drop=True)

        morningstar = morningstar.set_index(
            ['EffectiveDate', 'Ticker', 'Category', 'Item']
        )[['ItemValue', 'AssetType']]

        return morningstar


if __name__ == '__main__':
    l = LoadMorningStar()

    morningstar = l.load_morningstar_fund_attributes(
        end_date = date.today(), tickers=['VGPMX', 'VASVX']
    )

    morningstar.to_csv(
        "C:/Users/1/Documents/MySqlDB/LoadFiles/SecurityAttributes.csv"
    )


