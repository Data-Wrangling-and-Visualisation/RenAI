import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_csv(file_path: str) -> pd.DataFrame:
    """
    Loads a CSV file into a pandas DataFrame.
    """
    return pd.read_csv(file_path, encoding='utf-8')

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renames columns from CSV to standard names if necessary,
    and converts numeric fields to numbers.
    Adjust 'rename_dict' if your CSV columns differ.
    """
    
    rename_dict = {
        'artistDisplayName': 'Artist',
        'country': 'Country',
        'department': 'Department',
        'isPublicDomain': 'IsPublicDomain',
        'medium': 'Medium',
        'accessionYear': 'AccessionYear',
        'objectDate': 'ObjectDate'
    }
    

    for old_col, new_col in rename_dict.items():
        if old_col in df.columns:
            df.rename(columns={old_col: new_col}, inplace=True)

    if 'IsPublicDomain' not in df.columns:
        df['IsPublicDomain'] = False

    if 'ObjectDate' in df.columns:
        df['ObjectDate'] = pd.to_numeric(df['ObjectDate'], errors='coerce')
    if 'AccessionYear' in df.columns:
        df['AccessionYear'] = pd.to_numeric(df['AccessionYear'], errors='coerce')

    return df

def plot_distribution_by_year(df: pd.DataFrame):
    if 'ObjectDate' not in df.columns:
        print("No 'ObjectDate' column found in DataFrame.")
        return
    
    valid_years = df['ObjectDate'].dropna()
    if valid_years.empty:
        print("No valid ObjectDate data to plot.")
        return

    plt.figure(figsize=(10, 6))
    plt.hist(valid_years, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title('Distribution of Objects by Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Objects')
    plt.savefig('charts/distribution_by_year.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_medium_pie_chart(df: pd.DataFrame):
    if 'Medium' not in df.columns:
        print("No 'Medium' column found in DataFrame.")
        return
    
    medium_counts = df['Medium'].value_counts()
    if medium_counts.empty:
        print("No medium data to plot.")
        return

    top_mediums = medium_counts.head(10)

    fig, ax = plt.subplots(figsize=(8, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_mediums)))
    ax.pie(
        top_mediums,
        labels=top_mediums.index,
        autopct='%1.1f%%',
        startangle=140,
        colors=colors
    )
    ax.set_title('Distribution of Objects by Medium (Top 10)')
    plt.savefig('charts/distribution_by_medium.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_objects_by_artist(df: pd.DataFrame):
    if 'Artist' not in df.columns:
        print("No 'Artist' column found in DataFrame")
        return
    
    artist_counts = df['Artist'].value_counts()
    if artist_counts.empty:
        print("No artist data to plot.")
        return

    top_artists = artist_counts.head(10)

    plt.figure(figsize=(10, 6))
    y_pos = np.arange(len(top_artists))
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_artists)))

    plt.barh(y_pos, top_artists.values, color=colors, alpha=0.7)
    plt.yticks(y_pos, top_artists.index)
    plt.xlabel('Number of Objects')
    plt.ylabel('Artist')
    plt.title('Top 10 Artists by Number of Objects')
    plt.gca().invert_yaxis()
    plt.savefig('charts/objects_by_artist.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_objects_by_department(df: pd.DataFrame):
    if 'Department' not in df.columns:
        print("No 'Department' column found in DataFrame.")
        return
    
    dept_counts = df['Department'].value_counts()
    if dept_counts.empty:
        print("No department data to plot.")
        return

    plt.figure(figsize=(10, 6))
    x_pos = np.arange(len(dept_counts))
    colors = plt.cm.tab20(np.linspace(0, 1, len(dept_counts)))

    plt.bar(x_pos, dept_counts.values, color=colors, alpha=0.7)
    plt.xticks(x_pos, dept_counts.index, rotation=45, ha='right')
    plt.ylabel('Number of Objects')
    plt.title('Distribution of Objects by Department')
    plt.tight_layout()
    plt.savefig('charts/objects_by_department.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_objects_by_country(df: pd.DataFrame):
    if 'Country' not in df.columns:
        print("No 'Country' column found in DataFrame.")
        return
    
    country_counts = df['Country'].value_counts()
    if country_counts.empty:
        print("No country data to plot.")
        return

    top_countries = country_counts.head(10)

    plt.figure(figsize=(10, 6))
    y_pos = np.arange(len(top_countries))
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_countries)))

    plt.barh(y_pos, top_countries.values, color=colors, alpha=0.7)
    plt.yticks(y_pos, top_countries.index)
    plt.xlabel('Number of Objects')
    plt.ylabel('Country')
    plt.title('Top 10 Countries by Number of Objects')
    plt.gca().invert_yaxis()
    plt.savefig('charts/objects_by_country.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_public_domain_pie(df: pd.DataFrame):
    if 'IsPublicDomain' not in df.columns:
        print("No 'IsPublicDomain' column found in DataFrame.")
        return

    domain_counts = df['IsPublicDomain'].value_counts()
    if domain_counts.empty:
        print("No public domain data to plot.")
        return

    labels = ['Public Domain' if val else 'Not Public Domain' for val in domain_counts.index]
    colors = ['lightgreen', 'lightcoral']

    plt.figure(figsize=(6, 6))
    plt.pie(
        domain_counts,
        labels=labels,
        autopct='%1.1f%%',
        colors=colors,
        startangle=140,
        wedgeprops={'alpha': 0.7}
    )
    plt.title('Public Domain vs. Not Public Domain')
    plt.savefig('charts/public_domain_pie.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_accession_timeline(df: pd.DataFrame):
    if 'AccessionYear' not in df.columns:
        print("No 'AccessionYear' column found in DataFrame.")
        return
    
    valid_accession = df['AccessionYear'].dropna().astype(int)
    if valid_accession.empty:
        print("No valid 'AccessionYear' data to plot.")
        return

    year_counts = valid_accession.value_counts().sort_index()

    plt.figure(figsize=(10, 6))
    plt.plot(year_counts.index, year_counts.values, marker='o', linestyle='-', color='blue', alpha=0.7)
    plt.title('Objects Acquired by Year (AccessionYear)')
    plt.xlabel('Year')
    plt.ylabel('Number of Objects')
    plt.grid(True, alpha=0.3)
    plt.savefig('charts/accession_timeline.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    file_path = 'processed/processed_data.csv'
    df = load_csv(file_path)
    df = prepare_data(df)

    plot_distribution_by_year(df)
    plot_medium_pie_chart(df)
    plot_objects_by_artist(df)
    plot_objects_by_department(df)
    plot_objects_by_country(df)
    plot_public_domain_pie(df)
    plot_accession_timeline(df)

    print("All charts have been saved as PNG files.")

if __name__ == '__main__':
    main()
