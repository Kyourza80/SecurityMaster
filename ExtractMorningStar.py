import pandas as pd
import numpy as np
from datetime import date
from pandas.tseries.offsets import *
import requests
import json
from lxml import html
import re
import urllib2
from bs4 import BeautifulSoup
from MySqlConnect import MySqlConnect

pd.set_option('display.width', 150)
pd.set_option('display.max_columns', 30)


class ExtractMorningStar(object):
    @staticmethod
    def _page(url):
        page = urllib2.urlopen(url)
        soup = BeautifulSoup(page, 'html.parser')
        return soup

    @staticmethod
    def _fund_body(page):
        body = page.find('div', class_='r_bodywrap')
        return body

    def remove_non_ascii(self, text):
        return ''.join(i for i in text if ord(i) < 128)

    def _fund_name(self, ticker):
        page = self._page(self._url(ticker))

        if page.find('div', {'class': 'r_title'}):
            header = page.find('div', {'class': 'r_title'}).find('h1').text
        elif page.find('head'):
            header = page.find('head').find('title')
            header = header.text.split('Report')[0].strip()
        else:
            return None

        header = header.strip()

        header = self.remove_non_ascii(header)

        return header.encode('ascii')

    @staticmethod
    def _fund_ticker(page):
        ticker = page.find('div', {'class': 'r_title'})
        return ticker.find('span', {'class': 'gry'}).text.strip()

    # u'\u2014

    def _asset_allocation(self, ticker):
        soup = self._fund_body(
            self._page(
                self._url(
                    ticker)))
        aa_table = soup.find('table', {'class': 'r_table1 text2',
                                       'id': 'asset_allocation_tab'})

        thead = aa_table.find('thead')
        thead = [str(h.text.strip()) for h in
                 thead.find_all('th', {'class': 'str'})]
        thead = {thead[0]: thead}

        tbody = aa_table.find('tbody')

        rows = tbody.find_all('tr')
        hdata = []
        rdata = []
        for row in rows:
            h = row.find('th')
            if h:
                hdata.append(h.text)
                rdata.append(
                    [d.text for d in row.find_all('td')]
                )
        df = {}
        for idx, val in enumerate(hdata):
            df[val] = rdata[idx]

        df.update(thead)

        df = pd.DataFrame(df).set_index('Type').T

        for col in thead['Type']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].replace(np.nan, 0)

        df = df[[c for c in thead['Type'] if c not in ['Type']]]

        col_rename = {
            '% Net': 'Portfolio',
            '% Short': 'Short',
            '% Long': 'Long',
            'Bench-mark': 'Benchmark',
            'Cat Avg': 'Category Avg'
        }

        df['Table'] = 'Asset Allocation'

        return df.rename(columns=col_rename)[
            ['Portfolio', 'Benchmark', 'Category Avg', 'Table']
        ]

    def _equity_style(self, ticker):
        soup = self._fund_body(
            self._page(
                self._url(
                    ticker)))
        tables = soup.find(
            'div',
            {'id': 'equity_style_tab'})

        if tables is None:
            print "{} has no Equity Style Table".format(ticker)
            return None

        tables = tables.find_all('table')

        table_names = [i.find('caption').text.strip() for i in tables]

        col_rename = {
            '% of Portfolio': 'Portfolio',
            'Stock Portfolio': 'Portfolio'
        }

        table_data = []
        for tb_idx, tb in enumerate(tables):
            thead = [h.text.strip() for h in
                     tb.find('thead').find_all('th', {'class': 'str'})]
            key = thead[0]
            thead = {thead[0]: thead[1:]}

            tbody = tb.find('tbody')

            rows = tbody.find_all('tr')
            hdata = []
            rdata = []
            for row in rows:
                h = row.find('th')
                if h:
                    hdata.append(h.text)
                    rdata.append(
                        [d.text for d in row.find_all('td')]
                    )
            df = {}
            for idx, val in enumerate(hdata):
                df[val] = rdata[idx]

            df.update(thead)

            df = pd.DataFrame(df).set_index(key).T

            for col in thead[key]:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].replace(np.nan, 0)

            df['Table'] = table_names[tb_idx]
            df = df.rename(columns=col_rename)

            table_data.append(df)

        table_data = pd.concat(table_data)
        table_data['AssetType'] = 'Equity'

        table_data = table_data[
            ['Portfolio', 'Benchmark', 'Category Avg', 'Table', 'AssetType']]

        return table_data

    def _bond_style(self, ticker):
        soup = self._fund_body(
            self._page(
                self._url(
                    ticker)))
        tables = soup.find(
            'div',
            {'id': 'bond_style_tab'}
        )

        if tables is None:
            print "{} has no Bond Style Table".format(ticker)
            return None

        tables = tables.find_all('table')

        bad_names = ['Credit Quality']
        table_names = [
            str(i) for i in [i.find('caption').text.strip() for i in tables] if
            i not in bad_names]

        good_tables = []
        bad_tables = []
        for tbl in tables:
            if tbl.find(text='Credit Quality') is None:
                good_tables.append(tbl)
            else:
                bad_tables.append(tbl)

        table_data = []
        for tb_idx, tb in enumerate(good_tables):
            thead = [h.text.strip() for h in
                     tb.find('thead').find_all('th', {'class': 'str'})]
            key = thead[0]
            thead = {thead[0]: thead[1:]}

            tbody = tb.find('tbody')

            rows = tbody.find_all('tr')
            hdata = []
            rdata = []
            for row in rows:
                h = row.find('th')
                if h:
                    hdata.append(h.text)
                    rdata.append(
                        [d.text for d in row.find_all('td')]
                    )
            df = {}
            for idx, val in enumerate(hdata):
                df[val] = rdata[idx]

            df.update(thead)

            df = pd.DataFrame(df).set_index(key).T

            for col in thead[key]:
                df.loc[
                    df.index == 'Average Credit Quality', 'StringValue'
                ] = df[col]

                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].replace(np.nan, 0)

            df['Table'] = table_names[tb_idx]
            df = df.rename(columns={'% Bonds': 'Value'})
            # df = df.rename(columns=col_rename)

            table_data.append(df)

        for tb_idx, tb in enumerate(bad_tables):
            thead = [h.text.strip() for h in
                     tb.find('thead').find_all('th', {'class': 'str'})]
            key = thead[0]
            thead = {thead[0]: thead[1:4]}

            tbody = tb.find('tbody')

            rows = tbody.find_all('tr')

            hdata = []
            rdata = []
            for r in rows:
                if r.find('th'):
                    hdata.append(r.find('th').text)
                    rdata.append(
                        [d.text for d in r.find_all('td')[:3]]
                    )

            df = {}
            for idx, val in enumerate(hdata):
                df[val] = rdata[idx]

            df.update(thead)

            df = pd.DataFrame(df).set_index(key).T

            for col in thead[key]:
                df.loc[
                    df.index == 'Average Credit Quality', 'StringValue'
                ] = df[col]

                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].replace(np.nan, 0)

            df['Table'] = table_names[tb_idx]
            df = df.rename(columns={'% Bonds': 'Portfolio'})

            table_data.append(df)

        table_data = pd.concat(table_data)
        table_data.StringValue = table_data.StringValue.str.strip()

        table_data.loc[table_data.index.isin([
            'A', 'AA', 'AAA', 'B', 'BB', 'BBB', 'Below B', 'Not Rated'
        ]), 'Table'] = 'Credit Quality'

        table_data.Portfolio = table_data.Portfolio.combine_first(
            table_data.StringValue)
        table_data.Portfolio = table_data.Portfolio.combine_first(
            table_data.Value)

        table_data = table_data[
            table_data.Table.isin(
                ['Bond Statistics', 'Credit Quality']
            )]

        table_data['AssetType'] = 'Bond'

        table_data = table_data[
            ['Portfolio', 'Benchmark', 'Category Avg', 'Table', 'AssetType']]

        return table_data

    def _equity_sector_weightings(self, ticker):
        soup = self._fund_body(
            self._page(
                self._url(
                    ticker)))
        eq_tables = soup.find('div', {'id': 'sectorWeightings'}).find(
            'table', {'id': 'sector_we_tab'})

        if eq_tables is None:
            print "{} has no Equity Sector Table".format(ticker)
            return None

        table_head = ['% Stocks', 'Benchmark', 'Category Avg']
        table_head = {'Weights': table_head}

        tables = eq_tables.find('tbody')

        rows = tables.find_all('tr')

        hdata = []
        rdata = []
        for row in rows:
            if row.find('th') and len(row.find_all('td')) > 0:
                thead = row.find('th').text.strip()
                tbody = [i.text for i in row.find_all('td')]
                if tbody[0] == '':
                    tbody = tbody[1:]

                hdata.append(thead)
                rdata.append(tbody)

        df = {}
        for idx, val in enumerate(hdata):
            df[val] = rdata[idx][:3]

        df.update(table_head)

        df = pd.DataFrame(df).set_index('Weights').T

        df = df.rename(columns={'% Stocks': 'Portfolio'})

        for col in ['Portfolio', 'Benchmark', 'Category Avg']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        eq_sectors = [
            'Basic Materials', 'Consumer Cyclical', 'Financial Services',
            'Real Estate', 'Communication Services', 'Energy', 'Industrials',
            'Technology', 'Consumer Defensive', 'Healthcare', 'Utilities'
        ]

        df.loc[df.index.isin(eq_sectors), 'Table'] = 'Equity Sectors'

        df = df[df.Table.notnull()].sort_values('Table')

        df['AssetType'] = 'Equity'

        return df

    def _bond_sector_weightings(self, ticker):
        url = self._url(ticker)
        url = url + "#bond_sector_tab"
        soup = self._fund_body(
            self._page(url))

        bd_tables = soup.find('table', {'id': 'sector_wb_tab'})

        table_head = ['% Bonds', 'Benchmark', 'Category Avg']
        table_head = {'Type': table_head}

        tables = bd_tables

        if tables is None:
            print "{} has no Bond Sector Table".format(ticker)
            return None

        rows = tables.find_all('tr')

        hdata = []
        rdata = []
        for row in rows:
            if row.find('th') and len(row.find_all('td')) > 0:
                thead = row.find('th').text.strip()
                tbody = [i.text for i in row.find_all('td')]
                if tbody[0] == '':
                    tbody = tbody[1:]

                hdata.append(thead)
                rdata.append(tbody[:3])

        df = {}
        for idx, val in enumerate(hdata):
            df[val] = rdata[idx][:3]

        df.update(table_head)

        df = pd.DataFrame(df).set_index('Type').T

        df = df.rename(columns={'% Bonds': 'Portfolio'})

        for col in ['Portfolio', 'Benchmark', 'Category Avg']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        bd_sectors = [
            'Government', 'Government-Related', 'Corporate',
            'Agency Mortgage-Backed',
            'Non-Agency Residential MBS', 'Commercial MBS', 'Asset-Backed',
            'Covered Bond', 'Municipal', 'Cash & Equivalents', 'Other',
            'Securitized'
        ]

        df.loc[df.index.isin(bd_sectors), 'Table'] = 'Bond Sectors'

        df = df[
            (df.Table.notnull()) &
            (df.Portfolio.notnull())
            ].sort_values('Table')

        df['AssetType'] = 'Bond'

        return df

    def _bond_coupon(self, ticker):
        url = self._url(ticker)
        url = url + "#bond_sector_tab"
        soup = self._fund_body(
            self._page(url))

        bd_tables = soup.find('table', {'id': 'sector_wb_tab1'})

        table_head = ['% Bonds', 'Benchmark', 'Category Avg']
        table_head = {'Type': table_head}

        tables = bd_tables

        if tables is None:
            print "{} has no Bond Sector Table".format(ticker)
            return None

        rows = tables.find_all('tr')

        hdata = []
        rdata = []
        for row in rows:
            if row.find('th') and len(row.find_all('td')) > 0:
                thead = row.find('th').text.strip()
                tbody = [i.text for i in row.find_all('td')]
                if tbody[0] == '':
                    tbody = tbody[1:]

                hdata.append(thead)
                rdata.append(tbody[:3])

        df = {}
        for idx, val in enumerate(hdata):
            df[val] = rdata[idx][:3]

        df.update(table_head)

        df = pd.DataFrame(df).set_index('Type').T

        df = df.rename(columns={'% Bonds': 'Portfolio'})

        for col in ['Portfolio', 'Benchmark', 'Category Avg']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        coupons = [
            '0%', '0% to 4%', '4% to 6%', '6% to 8%', '8% to 10%',
            '10% to 12%', 'More than 12%'
        ]

        df.loc[df.index.isin(coupons), 'Table'] = 'Coupons'

        df = df[df.Table.notnull()].sort_values('Table')

        df['AssetType'] = 'Bond'

        return df

    def _bond_maturity(self, ticker):
        url = self._url(ticker)
        soup = self._fund_body(
            self._page(url))

        # headers = soup.find_all('h3')
        #
        # if tables is None:
        #     print "{} has no Bond Sector Table".format(ticker)
        #     return None

        maturity = [i for i in soup.find_all('h3') if
                    i.text.strip() in 'Bond Maturity Breakdown']

        if len(maturity) == 0:
            print "{} has no Bond Maturity Table".format(ticker)
            return None
        else:
            maturity = maturity[0]

        maturity = maturity.parent.parent

        table_head = ['% Bonds', 'Benchmark', 'Category Avg']
        table_head = {'Type': table_head}

        tables = maturity

        rows = tables.find_all('tr')

        hdata = []
        rdata = []
        for row in rows:
            if row.find('th') and len(row.find_all('td')) > 0:
                thead = row.find('th').text.strip()
                tbody = [i.text for i in row.find_all('td')]
                if tbody[0] == '':
                    tbody = tbody[1:]

                hdata.append(thead)
                rdata.append(tbody[:3])

        df = {}
        for idx, val in enumerate(hdata):
            df[val] = rdata[idx][:3]

        df.update(table_head)

        df = pd.DataFrame(df).set_index('Type').T

        df = df.rename(columns={'% Bonds': 'Portfolio'})

        for col in ['Portfolio', 'Benchmark', 'Category Avg']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df[df.Portfolio.notnull()]

        if df.shape[0] == 0:
            print "{} has empty Bond Maturity Table".format(ticker)
            return None

        maturity = [
            '1 to 3 Years', '10 to 15 Years', '15 to 20 Years',
            '20 to 30 Years', '3 to 5 Years', '5 to 7 Years',
            '7 to 10 Years', 'Over 30 Years',
        ]

        df.loc[df.index.isin(maturity), 'Table'] = 'Maturity'

        df = df[df.Table.notnull()].sort_values('Table')
        df['AssetType'] = 'Bond'

        return df

    def _bond_country_holding(self, ticker):
        url = self._url(ticker)
        soup = self._fund_body(
            self._page(url))

        country = [i for i in soup.find_all('h3') if
                   i.text.strip() in 'Top 10 Country Breakdown']

        if len(country) == 0:
            print "{} has no Bond Country Table".format(ticker)
            return None
        else:
            country = country[0]

        country = country.parent.parent.find('table')
        tables = country

        rows = tables.find_all('tr')
        table_head = ['Country', '% Bonds', 'Benchmark', 'Category Avg']

        rdata = []
        for row in rows:
            if len(row.find_all('td')) > 0:
                tbody = [i.text for i in row.find_all('td')]
                if tbody[0] == '':
                    tbody = tbody[1:]

                rdata.append(tbody[:4])

        rdata = [i for i in rdata if len(i) == 4]
        df = pd.DataFrame(rdata, columns=table_head)

        df = df.rename(columns={'% Bonds': 'Portfolio'})

        df = df[df.Portfolio.notnull()]

        df['Table'] = 'Country Breakdown'
        df['AssetType'] = 'Bond'

        return df.set_index('Country')

    def _equity_world_regions(self, ticker):
        soup = self._fund_body(self._page(self._url(ticker)))

        regions = soup.find('div', {'id': 'worldRegions'})

        if regions is None:
            "{} has no Equity Region Table".format(ticker)
            return None

        reg_table = regions.find('table', {'id': 'world_regions_tab'})

        reg_table = reg_table

        theader = reg_table.find('thead')
        theader = theader.find_all('th', {'class': 'str'})
        theader = {'Region': [str(i.text.strip()) for i in theader]}

        tables = reg_table.find('tbody')

        rows = tables.find_all('tr')

        hdata = []
        rdata = []

        region_names = [
            'Americas', 'North America', 'Latin America', 'Greater Europe',
            'United Kingdom', 'Europe Developed', 'Europ Emerging',
            'Africa/Middle East', 'Greater Asia', 'Japan', 'Australasia',
            'Asia Developed', 'Asia Emerging', '% Developed Markets',
            '% Emerging Markets', 'Australia'
        ]

        for row in rows:
            if len(row.find_all('td')) > 0:
                td = [i.text.strip() for i in row.find_all('td')]
                if len([h for h in td if h in region_names]) > 0:
                    thead = [h for h in td if h in region_names][0]
                    tbody = [h for h in td if h not in region_names]

                    hdata.append(thead)
                    rdata.append(tbody)

        df = {}
        for idx, val in enumerate(hdata):
            df[val] = rdata[idx][:3]
        df.update(theader)

        df = pd.DataFrame(df).set_index('Region').T
        df = df.rename(columns={'% Stocks': 'Portfolio'})
        for col in ['Portfolio', 'Benchmark', 'Category Avg']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        table_classification = {
            '% Developed Markets': 'Market Classification',
            '% Emerging Markets': 'Market Classification',
            'Americas': 'Super Region',
            'Greater Europe': 'Super Region',
            'Greater Asia': 'Super Region',
            'Asia Developed': 'World Region',
            'Asia Emerging': 'World Region',
            'Australasia': 'World Region',
            'Japan': 'World Region',
            'Europe Developed': 'World Region',
            'Latin America': 'World Region',
            'North America': 'World Region',
            'United Kingdom': 'World Region',
            'Africa/Middle East': 'World Region'
        }

        df = df.reset_index()
        for key, val in table_classification.iteritems():
            df.loc[df['index'] == key, 'Table'] = val
        df['AssetType'] = 'Equity'

        df = df[df['Table'] != 'Super Region']

        return df.set_index('index')

    def _update_date(self, ticker):
        soup = self._fund_body(self._page(self._url(ticker)))
        soup = soup.find('table', {'id': 'asset_allocation_tab'}).find('tfoot')
        values = soup.find(text=re.compile('As of'))
        values = values.replace('\*', '')[6:].strip()

        return values.encode('utf-8')

    def _url(self, ticker):
        base_url = 'http://portfolios.morningstar.com/fund/summary?'
        ticker = 't={}'.format(ticker)
        region = '&region=usa&culture=en-US'
        return base_url + ticker + region

    def _morningstar_webcall(self, **kwargs):
        tickers = kwargs['tickers']
        if tickers:
            if not isinstance(tickers, (list, tuple)):
                tickers = [tickers]

        master_list = []
        for index, ticker in enumerate(tickers):
            print 'processing: ', ticker

            url = 'http://portfolios.morningstar.com/fund/summary?t={}'.format(
                ticker)

            page = requests.get(url)
            tree = html.fromstring(page.content)

            '''
            Create Security headers
            '''
            sec_name = tree.xpath(
                '/html/body/div[1]/div[3]/div[1]/div/div[3]/h1/text()')
            as_of = tree.xpath(
                '//*[@id="asset_allocation_tab"]/tfoot/tr/td/text()')

            sec_name = self.remove_non_ascii(sec_name[0])

            sec_header = pd.DataFrame(
                dict(
                    SecurityName=sec_name,
                    AsOf=as_of[0],
                    Ticker=ticker
                ), index=[0])

            assets = []
            for i in range(2, 30):
                exec """sectype_{} = tree.xpath('//*[@id="asset_allocation_tab"]/tbody/tr[{}]/th/text()')""".format(
                    i, i)

                if len(eval('sectype_{}'.format(i))) > 0:
                    assets.append(eval('sectype_{}'.format(i)))

            assets = pd.DataFrame(assets, columns=['Column'])

            asset_weights = []
            for i in range(2, 30):
                exec """value_{} = tree.xpath('//*[@id="asset_allocation_tab"]/tbody/tr[{}]/td[2]/text()')""".format(
                    i, i)

                if len(eval('value_{}'.format(i))) == 1:
                    asset_weights.append(eval('value_{}'.format(i)))

            asset_weights = pd.DataFrame(asset_weights, columns=['Weight'])

            if assets.shape[0] > 0 and asset_weights.shape[0] > 0:
                asset_class = pd.concat([assets, asset_weights], axis=1)

                asset_class['Category'] = 'AssetClass'

            else:
                asset_class = pd.DataFrame(
                    columns=['Column', 'Weight', 'Category'])

            cr_ratings = []
            for i in range(2, 30):
                exec """cr_field_{} = tree.xpath('//*[@id="bond_style_tab"]/div/table/tbody/tr[{}]/th/text()')""".format(
                    i, i)

                if len(eval('cr_field_{}'.format(i))) > 0:
                    cr_ratings.append(eval('cr_field_{}'.format(i)))

            cr_ratings = pd.DataFrame(cr_ratings, columns=['Column'])

            cr_weights = []
            for i in range(2, 30):
                exec """cr_val_{} = tree.xpath('//*[@id="bond_style_tab"]/div/table/tbody/tr[{}]/td[1]/text()')""".format(
                    i, i)

                if len(eval('cr_val_{}'.format(i))) == 1:
                    cr_weights.append(eval('cr_val_{}'.format(i)))

            cr_weights = pd.DataFrame(cr_weights, columns=['Weight'])

            if cr_ratings.shape[0] > 0 and cr_weights.shape[0] > 0:

                credit_ratings = pd.concat([cr_ratings, cr_weights], axis=1)

                credit_ratings.Weight = pd.to_numeric(
                    credit_ratings.Weight, errors=coerce
                )

                credit_ratings = credit_ratings[
                    credit_ratings.Weight.notnull()
                ]

                credit_ratings['Category'] = 'CreditRating'

            else:
                credit_ratings = pd.DataFrame(
                    columns=['Column', 'Weight', 'Category'])

            ig = ['AAA', 'AA', 'A', 'BBB']
            junk = ['BB', 'B', 'Below B', 'Not Rated']

            credit = credit_ratings.copy()

            credit.loc[credit.Column.isin(ig), 'Column'] = 'InvestmentGrade'

            credit.loc[credit.Column.isin(junk), 'Column'] = 'JunkBond'

            credit = credit.groupby('Column').Weight.sum().reset_index()

            credit['Category'] = 'Aggregate CreditRating'

            credit_ratings = pd.concat([credit_ratings, credit])

            mat_dates = []
            for i in range(2, 30):
                exec """mat_field_{} = tree.xpath('//*[@id="sectorWeightings"]/div[5]/table/tbody/tr[{}]/th/text()')""".format(
                    i, i)

                if len(eval('mat_field_{}'.format(i))) > 0:
                    mat_dates.append(eval('mat_field_{}'.format(i)))

            mat_dates = pd.DataFrame(mat_dates, columns=['Column'])

            mat_weights = []
            for i in range(2, 30):
                exec """mat_val_{} = tree.xpath('//*[@id="sectorWeightings"]/div[5]/table/tbody/tr[{}]/td[1]/text()')""".format(
                    i, i)

                if len(eval('mat_val_{}'.format(i))) == 1:
                    mat_weights.append(eval('mat_val_{}'.format(i)))

            mat_weights = pd.DataFrame(mat_weights, columns=['Weight'])

            if mat_dates.shape[0] > 0 and mat_weights.shape[0] > 0:
                maturities = pd.concat([mat_dates, mat_weights], axis=1)

                maturities['Category'] = 'Maturity'

                maturities.Weight = pd.to_numeric(
                    maturities.Weight, errors=coerce)

                maturities = maturities[maturities.Weight.notnull()]

            else:
                maturities = pd.DataFrame(
                    columns=['Column', 'Weight', 'Category'])

            short = ['1 to 3 Years']

            intermediate = ['3 to 5 Years', '5 to 7 Years', '7 to 10 Years']

            long = ['10 to 15 Years', '15 to 20 Years', '20 to 30 Years',
                    'Over 30 Years']

            mat = maturities.copy()

            mat.loc[mat.Column.isin(short), 'Column'] = 'ShortTerm'

            mat.loc[mat.Column.isin(intermediate), 'Column'] = 'MediumTerm'

            mat.loc[mat.Column.isin(long), 'Column'] = 'LongTerm'

            mat = mat.groupby('Column').Weight.sum().reset_index()
            mat['Category'] = 'Aggregate Maturity'

            maturities = pd.concat([maturities, mat])

            caps = []
            for i in range(2, 30):
                exec """style_field_{} = tree.xpath('//*[@id="equity_style_tab"]/tbody/tr[{}]/th/text()')""".format(
                    i, i)

                if len(eval('style_field_{}'.format(i))) > 0:
                    caps.append(eval('style_field_{}'.format(i)))

            caps = pd.DataFrame(caps, columns=['Column'])

            cap_weights = []
            for i in range(2, 30):
                exec """style_val_{} = tree.xpath('//*[@id="equity_style_tab"]/tbody/tr[{}]/td[1]/text()')""".format(
                    i, i)

                if len(eval('style_val_{}'.format(i))) == 1:
                    cap_weights.append(eval('style_val_{}'.format(i)))

            cap_weights = pd.DataFrame(cap_weights, columns=['Weight'])

            if caps.shape[0] > 0 and cap_weights.shape[0] > 0:
                market_caps = pd.concat([caps, cap_weights], axis=1)
                market_caps['Category'] = 'MarketCap'
            else:
                market_caps = pd.DataFrame(
                    columns=['Column', 'Weight', 'Category'])

            regions = []
            for i in range(2, 40):
                exec """reg_field_{} = tree.xpath('//*[@id="world_regions_tab"]/tbody/tr[{}]/td[1]/text()')""".format(
                    i, i)

                if len(eval('reg_field_{}'.format(i))) > 0:
                    regions.append(eval('reg_field_{}'.format(i)))

            regions = pd.DataFrame(regions, columns=['Column'])
            regions = regions[
                regions.Column != 'Market Classification'
                ].reset_index(drop=True)
            regions.Column = regions.Column.str.replace('% ', '')

            reg_weights = []
            for i in range(2, 40):
                exec """reg_val_{} = tree.xpath('//*[@id="world_regions_tab"]/tbody/tr[{}]/td[2]/text()')""".format(
                    i, i)

                if len(eval('reg_val_{}'.format(i))) == 1:
                    reg_weights.append(eval('reg_val_{}'.format(i)))

            reg_weights = pd.DataFrame(reg_weights, columns=['Weight'])

            if regions.shape[0] > 0 and reg_weights.shape[0] > 0:

                country_regions = pd.concat([regions, reg_weights], axis=1)

                country_regions['Category'] = 'Region'

                country_regions.loc[country_regions.Column.isin([
                    'Americas', 'Greater Europe', 'Greater Asia'
                ]), 'Category'] = 'Aggregate Region'

                country_regions.loc[country_regions.Column.isin(
                    ['Developed Markets', 'Emerging Markets']
                ), 'Category'] = 'Market Region'

            else:
                country_regions = pd.DataFrame(
                    columns=['Column', 'Weight', 'Column'])

            sec_dir = {'Ticker': ticker, 'sec_name': sec_name, 'as_of': as_of}

            combined = []
            for i in ['asset_class', 'credit_ratings', 'maturities',
                      'market_caps', 'country_regions']:
                if eval(i).shape[0] > 0:
                    eval(i).Weight = pd.to_numeric(
                        eval(i).Weight, errors=coerce
                    )
                    combined.append(eval(i))

            combined = pd.concat(combined)

            combined = combined[combined.Weight.notnull()]

            for i in list(combined.Column):
                combined.loc[
                    combined.Column == i, 'Column'] = self.remove_non_ascii(i)

            combined['Weight'] = combined.Weight.astype(float)

            validate = combined.groupby('Category').Weight.sum().reset_index()

            validate['PctWeight'] = validate['Weight'] / 100

            if validate[validate.Weight.round(0) != 100].shape[0] > 0:
                print validate[
                    validate.Weight.round(2) != 100
                    ].rename(columns={'Category': ticker})

            combined = pd.merge(
                combined,
                validate[validate.Category != 'AssetClass'][
                    ['Category', 'PctWeight']],
                'left', 'Category')

            combined['Weight'] = combined.Weight / combined['PctWeight'].fillna(
                1)
            combined = combined.drop('PctWeight', axis=1)

            premaster = combined

            premaster['Index'] = 0

            premaster = pd.pivot_table(
                premaster, values='Weight', index='Index', columns='Column'
            ).reset_index(drop=True).rename_axis(None, axis=1)

            premaster['Ticker'] = sec_dir['Ticker']
            premaster['as_of'] = sec_dir['as_of']
            premaster['sec_name'] = sec_dir['sec_name']

            master_list.append(premaster)

            print "{}: complete".format(ticker)

        master_df = pd.concat(master_list)

        master_df['as_of'] = master_df.as_of.str[6:]

        normalized = kwargs['normalized']

        if normalized:
            index_cols = ['Ticker', 'as_of', 'sec_name']
            attribute_cols = [str(i) for i in master_df.columns if
                              i not in index_cols]

            master_df = pd.melt(
                master_df,
                id_vars=index_cols,
                value_vars=attribute_cols,
                var_name='Attribute',
                value_name='AttributeValue'
            )

            master_df = master_df[master_df.AttributeValue.notnull()]

            master_df = master_df.rename(
                columns={
                    'as_of': 'EffectiveDate',
                    'sec_name': 'SecurityName'
                }).sort_values(
                ['Ticker', 'EffectiveDate', 'Attribute']
            ).reset_index(drop=True)

        return master_df

    def morningstar_beta(self,
                         tickers=None,
                         savecsv='security_description',
                         saveraw=False,
                         sqlcnx=True
                         ):
        """
        Get Security Descriptions for mutual funds and ETFs.
        Saving options:
            1. raw file: all columns from morningstar including bond desciptions
            2. formatted file with equity geo, cap, and asset allocations.
                used to for mysql import

        :param tickers:
        :param savecsv:
        :param saveraw:
        :param sqlcnx:
        :return:
        """

        master_df = self._morningstar_webcall(tickers=tickers, normalized=False)

        '''
        Master Data Frame formatting
        Type Formating
        '''
        numeric_columns = [
            '1 to 3 Years', '10 to 15 Years', '15 to 20 Years',
            '20 to 30 Years',
            '3 to 5 Years', '5 to 7 Years', '7 to 10 Years', 'ShortTerm',
            'MediumTerm', 'LongTerm', 'A', 'AA', 'AAA', 'Africa/Middle East',
            'InvestmentGrade', 'JunkBond', 'Americas', 'Asia Developed',
            'Asia Emerging', 'Australasia', 'B', 'BB', 'BBB', 'Below B', 'Bond',
            'Cash', 'Europe Developed', 'Europe Emerging', 'Giant',
            'Greater Asia', 'Greater Europe', 'Japan', 'Large', 'Latin America',
            'Medium', 'Micro', 'Non US Stock', 'North America', 'Not Rated',
            'Other', 'Over 30 Years', 'Small', 'US Stock', 'United Kingdom'
        ]

        missing_cols = [
            col for col in numeric_columns if col not in list(master_df.columns)
        ]

        for col in missing_cols:
            master_df[col] = None

        master_df[['North America', 'Americas']] = master_df[
            ['North America', 'Americas']].astype(float)

        '''
        if not all at columns are present
        '''
        col_check = pd.DataFrame(numeric_columns, columns=['cols'])

        col_check = col_check[col_check.cols.isin(list(master_df))].cols

        numeric_columns = list(col_check)

        master_df = master_df.fillna(0)

        for i in numeric_columns:
            master_df[i] = pd.to_numeric(master_df[i], errors='coerce')

        master_df['am_dm'] = master_df['North America']

        master_df['am_em'] = master_df['Americas'] - master_df['North America']

        master_df['eu_dm'] = master_df['United Kingdom'] + master_df[
            'Europe Developed']

        master_df['eu_em'] = master_df['Europe Emerging'] + master_df[
            'Africa/Middle East']

        master_df['ap_dm'] = master_df['Greater Asia'] - master_df[
            'Asia Emerging']

        master_df['ap_em'] = master_df['Asia Emerging']

        master_df['Term_Short'] = master_df['1 to 3 Years'] + master_df[
            '3 to 5 Years']

        master_df['Term_Intermediate'] = master_df['5 to 7 Years'] + master_df[
            '7 to 10 Years']

        master_df['Term_Long'] = master_df['10 to 15 Years'] + master_df[
            '15 to 20 Years'] + master_df['20 to 30 Years'] + master_df[
                                     'Over 30 Years']

        master_df['Grade_Investment'] = master_df['AAA'] + master_df[
            'AA'] + master_df['A'] + master_df['BBB']

        master_df['Grade_Speculative'] = master_df['BB'] + master_df[
            'B'] + master_df['Below B']

        if saveraw:
            save_path = r"C:\Users\1\Documents\python_scripts\csv_output\raw_descriptive.csv"
            master_df.set_index(
                ['as_of', 'Ticker']).to_csv(save_path, encoding='utf-8')

        smf_stmt = """
                   select Ticker, SecurityType, VanguardCode
                   from portfolio.securitymaster
                   where EndDate is null
               """
        if sqlcnx:
            ms = MySqlConnect()
            sectype_df = ms.mysql_getdata(smf_stmt)

            master_df = pd.merge(master_df, sectype_df, on='Ticker', how='left')
        else:
            master_df['SecurityType'] = None
            master_df['VanguardCode'] = None

        equity_result = master_df[
            ['as_of', 'Ticker', 'sec_name', 'SecurityType', 'Bond', 'US Stock',
             'Non US Stock', 'Other', 'am_dm', 'am_em', 'eu_dm', 'eu_em',
             'ap_dm', 'ap_em', 'Giant', 'Large', 'Medium', 'Small', 'Micro',
             'VanguardCode']]

        # if savecsv is not None:
        #     save_path = r"C:\Users\1\Documents\MySqlDB\{}.csv".format(
        #         savecsv)
        #     equity_result.set_index(['as_of', 'Ticker']).to_csv(
        #         save_path, encoding='utf-8')
        #     # , line_terminator='\r\n')

        return equity_result

        # master_df.to_csv('out.csv', encoding='utf-8')

    def morningstar_security(self, tickers):
        if not isinstance(tickers, (list, tuple)):
            tickers = [tickers]

        securities = self._morningstar_webcall(tickers=tickers, normalized=True)

        securities = securities[securities.AttributeValue.fillna(0) != 0]

        return securities

    def alphavantage_prices(self,
                            tickers,
                            outputsize='compact'
                            ):

        col_map = {
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '4. close': 'Close',
            '5. adjusted close': 'AdjustedClose',
            '6. volume': 'Volume',
            '7. dividend amount': 'DividendAmount',
            '8. split coefficient': 'SplitCoefficient',
            'index': 'CloseDate'
        }

        if not isinstance(tickers, (list, tuple)):
            tickers = [tickers]

        prc_df = []
        failed_tickers = []
        for ticker in tickers:
            print 'Processing ', ticker
            try:
                prices = pd.DataFrame(self.ts.get_daily_adjusted(
                    symbol=ticker, outputsize=outputsize
                )[0]).T.reset_index()

                prices = prices.rename(columns=col_map)[
                    ['CloseDate', 'Close', 'AdjustedClose', 'DividendAmount',
                     'SplitCoefficient']]

                prices['Ticker'] = ticker

                prc_n = []
                for col in ['Close', 'AdjustedClose', 'DividendAmount',
                            'SplitCoefficient']:
                    df = prices[
                        ['CloseDate', 'Ticker', col]].rename(
                        columns={col: 'Value'})

                    df['Item'] = col

                    prc_n.append(df)

                prc_n = pd.concat(prc_n)

                prc_n = prc_n[
                    ['CloseDate', 'Item', 'Value', 'Ticker']
                ].sort_values(['Ticker', 'CloseDate', 'Item'])

                prc_n.Value = pd.to_numeric(prc_n.Value)

                prc_n.loc[
                    (prc_n.Item == 'DividendAmount') & (prc_n.Value == 0),
                    'Value'] = np.nan

                prc_n.loc[
                    (prc_n.Item == 'SplitCoefficient') & (prc_n.Value == 1),
                    'Value'] = np.nan

                prc_n = prc_n[prc_n.Value.notnull()]

                prc_df.append(prc_n)
            except:
                failed_tickers.extend([ticker])
                print 'fail'

        prc_df = pd.concat(prc_df)

        result = [prc_df, failed_tickers]

        return result


if __name__ == '__main__':
    ms = MySqlConnect()
    mstr = ExtractMorningStar()
    end_date = date.today()
    start_date = date.strftime(
        MonthBegin().rollback(end_date - DateOffset(months=1)),
        '%Y-%m-%d'
    )
    end_date = date.strftime(end_date, '%Y-%m-%d')

    tickers = ms.get_tickers(
        startdate=start_date, enddate=end_date, fundsonly=True
    )
    ticker = 'VWIAX'

    print mstr._asset_allocation(ticker).columns
    print mstr._equity_sector_weightings(ticker).columns
    print mstr._equity_world_regions(ticker).columns

    print mstr._bond_country_holding(ticker).columns
    print mstr._bond_maturity(ticker).columns
    print mstr._bond_coupon(ticker).columns
    print mstr._bond_sector_weightings(ticker).columns

    print mstr._bond_style(ticker).columns
    print mstr._equity_style(ticker).columns
    print mstr._equity_style(ticker)
