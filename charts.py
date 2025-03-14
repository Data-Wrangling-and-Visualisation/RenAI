import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json

# Load JSON file
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# Convert JSON to DataFrame
def json_to_dataframe(data):
    # We'll extract relevant fields; if some fields are missing, default to None or 'Unknown'
    records = []
    for item in data:
        record = {
            'AccessionYear': item.get('accessionYear'),
            'ObjectDate': item.get('objectDate'),
            'IsPublicDomain': item.get('isPublicDomain', False),
            'Medium': item.get('medium', 'Unknown'),
            'Artist': item.get('artistDisplayName', 'Unknown'),
            'Department': item.get('department', 'Unknown'),
            'Country': item.get('country', 'Unknown')
        }
        records.append(record)
    df = pd.DataFrame(records)

    # Convert numeric columns to numbers if needed
    # For ObjectDate and AccessionYear, let's coerce to numeric if possible
    df['ObjectDate'] = pd.to_numeric(df['ObjectDate'], errors='coerce')
    df['AccessionYear'] = pd.to_numeric(df['AccessionYear'], errors='coerce')

    return df

# 1) Histogram: Distribution of objects by ObjectDate
def plot_distribution_by_year(df):
    # Filter out NaN
    valid_years = df['ObjectDate'].dropna()
    if valid_years.empty:
        print("No valid ObjectDate data to plot.")
        return

    plt.figure(figsize=(10, 6))
    plt.hist(valid_years, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title('Distribution of Objects by Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Objects')
    plt.savefig('distribution_by_year.png', dpi=300, bbox_inches='tight')
    plt.close()

# 2) Pie chart: Distribution of objects by Medium (Top 10)
def plot_medium_pie_chart(df):
    medium_counts = df['Medium'].value_counts()
    if medium_counts.empty:
        print("No medium data to plot.")
        return

    # Take top 10 mediums
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
    plt.savefig('distribution_by_medium.png', dpi=300, bbox_inches='tight')
    plt.close()

# 3) Bar chart: Number of objects by Artist (Top 10)
def plot_objects_by_artist(df):
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
    plt.gca().invert_yaxis()  # invert y-axis so the first bar is at the top
    plt.savefig('objects_by_artist.png', dpi=300, bbox_inches='tight')
    plt.close()

# 4) Bar chart: Distribution by Department
def plot_objects_by_department(df):
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
    plt.savefig('objects_by_department.png', dpi=300, bbox_inches='tight')
    plt.close()

# 5) Bar chart: Number of objects by Country (Top 10)
def plot_objects_by_country(df):
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
    plt.savefig('objects_by_country.png', dpi=300, bbox_inches='tight')
    plt.close()

# 6) Pie chart: Public domain vs. not public domain
def plot_public_domain_pie(df):
    if 'IsPublicDomain' not in df.columns:
        print("No IsPublicDomain field in data.")
        return

    domain_counts = df['IsPublicDomain'].value_counts()

    labels = ['Public Domain' if val else 'Not Public Domain' for val in domain_counts.index]
    colors = ['lightgreen', 'lightcoral']  # specify some distinct colors

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
    plt.savefig('public_domain_pie.png', dpi=300, bbox_inches='tight')
    plt.close()

# 7) Line chart: Number of acquired objects by AccessionYear
# We interpret "AccessionYear" as the year the object was acquired or recorded.
# We'll group by year and count.
def plot_accession_timeline(df):
    # Filter out NaN and convert to int
    valid_accession = df['AccessionYear'].dropna().astype(int)
    if valid_accession.empty:
        print("No valid AccessionYear data to plot.")
        return

    # Count how many objects per year
    year_counts = valid_accession.value_counts().sort_index()

    plt.figure(figsize=(10, 6))
    plt.plot(year_counts.index, year_counts.values, marker='o', linestyle='-', color='blue', alpha=0.7)
    plt.title('Objects Acquired by Year (AccessionYear)')
    plt.xlabel('Year')
    plt.ylabel('Number of Objects')
    plt.grid(True, alpha=0.3)
    plt.savefig('accession_timeline.png', dpi=300, bbox_inches='tight')
    plt.close()

# Main function
def main():
    file_path = 'processed_data.json'  # Replace with your JSON file path
    data = load_data(file_path)
    df = json_to_dataframe(data)

    # Create all suggested visualizations
    plot_distribution_by_year(df)
    plot_medium_pie_chart(df)
    plot_objects_by_artist(df)
    plot_objects_by_department(df)
    plot_objects_by_country(df)
    plot_public_domain_pie(df)
    plot_accession_timeline(df)

    print("All charts have been saved as PNG files.")

if __name__ == "__main__":
    main()