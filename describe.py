#!/usr/bin/python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
from settings import CSV_TRAIN, CSV_TEST, CSV_STORE

# Load the data into a DataFrame
train = pd.read_csv(CSV_TRAIN, low_memory=False)
test = pd.read_csv(CSV_TEST, low_memory=False)
store = pd.read_csv(CSV_STORE, low_memory=False)

train = train[train['Sales'] != 0]


def summarize(df):
    print(train.describe())
    print('\n%13s %15s %s' % ('Column', 'Type', 'Unique Values'))
    print('-----------------------------------------------------')
    for col in df:
        print('%13s %15s %s' % (col, df[col].dtype, df[col].unique()))
    return


def sales_by_day():
    fig, (ax1) = plt.subplots(1, 1, figsize=(15, 2))
    sns.barplot(x='DayOfWeek', y='Sales', data=train, order=[1, 2, 3, 4, 5, 6, 7],
                ax=ax1, palette="hls")
    ax1.set_title("Average Sales By Day")
    plt.show()


def sales_by_store():
    fig, (ax1) = plt.subplots(1, 1, figsize=(20, 6))
    data = []
    grouped = train.groupby('Store')
    for key, group in grouped:
        data.append([key, group['Sales']])
    sorted_data = data[:50]
    sorted_data.sort(key=lambda x: np.median(x[1]))
    s_x = [sorted_data[i][1] for i in range(50)]
    s_index = [sorted_data[i][0] for i in range(50)]
    plt.boxplot(s_x, manage_xticks=True, labels=s_index)
    plt.title("Sales By Store")
    plt.xlabel("Store ID")
    plt.ylabel("Sales")
    plt.show()
    return


def sales_by_state_holiday():
    fig, (ax2) = plt.subplots(1, 1, figsize=(10, 6))
    sns.barplot(x='StateHoliday', y='Sales', data=train, order=['0', 'a', 'c'],
                ax=ax2, palette="hls")
    ax2.set_title("Average Sales Depending On State Holiday")
    ax2.set_xlabel("Type of State Holiday")
    ax2.set_ylabel("Mean Sales")
    plt.show()


def sales_by_school_holiday():
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
    sns.countplot(x='SchoolHoliday', data=train, ax=ax1, palette="hls")
    ax1.set_title("Training Data On School Holidays")
    ax1.set_xlabel("Is School Holiday")
    ax1.set_ylabel("Number of Datapoints")
    for p in ax1.patches:
        x = p.get_bbox().get_points()[:, 0]
        y = p.get_bbox().get_points()[1, 1]
        ax1.annotate('{:.0f}'.format(y), (x.mean(), y),
                     ha='center', va='bottom')

    sns.barplot(x='SchoolHoliday', y='Sales', data=train, ax=ax2, palette="hls")
    ax2.set_title("Average Sales Depending On School Holiday")
    ax2.set_xlabel("Is School Holiday")
    ax2.set_ylabel("Mean Sales")

    sns.countplot(x='StateHoliday', data=train, ax=ax3, palette="hls")
    ax3.set_title("Training Data On State Holidays")
    ax3.set_xlabel("Type of State Holiday")
    ax3.set_ylabel("Number of Datapoints")
    for p in ax3.patches:
        x = p.get_bbox().get_points()[:, 0]
        y = p.get_bbox().get_points()[1, 1]
        ax3.annotate('{:.0f}'.format(y), (x.mean(), y),
                     ha='center', va='bottom')

    sns.barplot(x='StateHoliday', y='Sales', data=train, order=['0', 'a', 'c'],
                ax=ax4, palette="hls")
    ax4.set_title("Average Sales Depending On State Holiday")
    ax4.set_xlabel("Type of State Holiday")
    ax4.set_ylabel("Mean Sales")

    plt.title("Effect of Holidays on Sales")
    plt.show()


def sales_by_store_by_promo():
    grouped = train.groupby(['Store', 'Promo'])
    no_promo = []
    promo = []
    for key, group in grouped:
        if key[1]:
            promo.append(group["Sales"].mean())
        else:
            no_promo.append(group["Sales"].mean())
    difference = [x - y for (x, y) in zip(promo, no_promo)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    ax1.boxplot([promo, no_promo], manage_xticks=True, labels=["Yes", "No"], vert=True)
    ax1.set_title("Average Sales of Stores With And Without Promo")
    ax1.set_ylabel("Has Promo")
    ax1.set_xlabel("Mean Sales of Store")

    sns.distplot(a=difference, kde=False, norm_hist=False, ax=ax2)
    ax2.set_title("Difference in Sales With and Without Promo By Store")
    ax2.set_ylabel("Number of Stores")
    ax2.set_xlabel("Difference in Mean Sales (Promo - Non-Promo)")
    plt.show()


def sales_by_christmas():
    today = pd.to_datetime(train['Date'])
    christmas = pd.Datetime(n)


def customer_to_sales():
    fig, (ax1) = plt.subplots(1, 1, figsize=(10, 10))
    sns.regplot(x='Customers', y='Sales', data=train)
    ax1.set_title("Relationship between Customers and Sales")
    ax1.set_xlabel("Number of Customers")
    ax1.set_ylabel("Sales")
    plt.show()


sales_by_school_holiday()