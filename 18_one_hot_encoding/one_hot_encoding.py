import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import logging

# تنظیمات لاگ‌گیری
logging.basicConfig(level=logging.INFO, format="%(message)s")


def load_data(file_path):
    """
    Load the dataset from a CSV file.
    Args:
        file_path: Path to the CSV file
    Returns:
        DataFrame containing the dataset
    """
    logging.info(f"Loading data from {file_path}...")
    return pd.read_csv(file_path)


def one_hot_encode(df, categorical_columns):
    """
    Perform One-Hot Encoding on specified categorical columns.
    Args:
        df: Original DataFrame
        categorical_columns: List of categorical columns to encode
    Returns:
        Transformed DataFrame with One-Hot Encoded columns
    """
    logging.info(f"One-Hot Encoding columns: {categorical_columns}")
    encoder = OneHotEncoder(sparse_output=False)  # Use dense array output
    encoded_array = encoder.fit_transform(df[categorical_columns])

    # Create a DataFrame for the encoded data
    encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(categorical_columns))
    logging.info(f"Encoded DataFrame shape: {encoded_df.shape}")
    logging.info(f"Categories: {encoder.categories_}")

    return encoded_df


def merge_and_clean(df, encoded_df, drop_columns):
    """
    Merge the encoded DataFrame with the original DataFrame and drop specified columns.
    Args:
        df: Original DataFrame
        encoded_df: Encoded DataFrame
        drop_columns: List of columns to drop
    Returns:
        Cleaned DataFrame
    """
    logging.info(f"Merging and dropping columns: {drop_columns}")
    merged_df = pd.concat([df, encoded_df], axis=1)
    cleaned_df = merged_df.drop(columns=drop_columns)
    return cleaned_df


def save_to_csv(df, file_path):
    """
    Save the DataFrame to a CSV file.
    Args:
        df: DataFrame to save
        file_path: Path to save the CSV file
    """
    logging.info(f"Saving processed data to {file_path}...")
    df.to_csv(file_path, index=False)


def main():
    """
    Main function to execute the entire pipeline.
    """
    # Load data
    file_path = 'house_prices.csv'
    df = load_data(file_path)

    # Specify categorical columns to encode
    categorical_columns = ['Brick', 'Neighborhood']

    # Perform One-Hot Encoding
    encoded_df = one_hot_encode(df, categorical_columns)

    # Merge and clean the DataFrame
    drop_columns = categorical_columns  # Drop original categorical columns
    cleaned_df = merge_and_clean(df, encoded_df, drop_columns)

    # Save the processed DataFrame to a new CSV file
    output_file = 'processed_house_prices.csv'
    save_to_csv(cleaned_df, output_file)
    logging.info("Processing completed successfully.")


if __name__ == "__main__":
    main()