import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


class HousePricePredictor:
    def __init__(self, data_path, binary_cols, numeric_cols, target_col, test_size=0.2, random_state=42):
        self.data_path = data_path
        self.binary_cols = binary_cols
        self.numeric_cols = numeric_cols
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state
        self.model = LinearRegression()
        self.modelReg=LogisticRegression(max_iter=2000,solver='lbfgs')
        self.features_train = None
        self.features_test = None
        self.target_train = None
        self.target_test = None
        self.feature_names = None

    def load_data(self):
        """Load and preview data"""
        self.df = pd.read_csv(self.data_path)
        print("Initial data sample:")
        print(self.df.head())

    def preprocess_data(self):
        """Preprocess data including binary encoding and one-hot encoding"""
        # Binary encoding for yes/no columns
        for col in self.binary_cols:
            self.df[col] = self.df[col].map({'yes': 1, 'no': 0})

        # One-hot encoding for furnishing status
        self.df = pd.get_dummies(self.df, columns=['furnishingstatus'], drop_first=True)
        print("\nData after one-hot encoding:")
        print(self.df.head())

    def remove_outliers(self):
        """Remove outliers using IQR method for numeric columns"""
        for col in self.numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]

        print("\nData after outlier removal:")
        print(self.df.head())

    def split_data(self):
        """Split data into training and testing sets"""
        features = self.df.drop(self.target_col, axis=1)
        target = self.df[self.target_col]

        self.feature_names = features.columns.tolist()
        (self.features_train,
         self.features_test,
         self.target_train,
         self.target_test) = train_test_split(features, target,
                                              test_size=self.test_size,
                                              random_state=self.random_state)



    def train_model(self):
        """Train the linear regression model"""
        self.model.fit(self.features_train, self.target_train)
        # print(f"\nModel coefficients: {self.model.coef_}")
        # print(f"Model intercept: {self.model.intercept_}")

        self.modelReg.fit(self.features_train, self.target_train)
        # print(f"\nModelReg coefficients: {self.modelReg.coef_}")
        # print(f"ModelReg intercept: {self.modelReg.intercept_}")

    def evaluate_model(self):
        """Evaluate model performance"""
        y_pred = self.model.predict(self.features_test)
        mse = mean_squared_error(self.target_test, y_pred)
        std = self.target_test.std()

        print(f"\nMean Squared Error: {mse}")
        print(f"Root MSE: {np.sqrt(mse)}")
        print(f"Target standard deviation: {std}")
        print(f'describe: {self.target_test.describe()}')

    def evaluate_model_reg(self):
        y_pred_reg=self.modelReg.predict(self.features_test)
        mse_reg=mean_squared_error(self.target_test, y_pred_reg)
        std_reg=self.target_test.std()
        print(f"\nREGLOSTIC Mean Squared Error: {mse_reg}")
        print(f"Root MSE: {np.sqrt(mse_reg)}")
        print(f"Target standard deviation: {std_reg}")
        print(f'describe: {self.target_test.describe()}')

    def prepare_new_data(self, new_data):
        """Prepare new data for prediction"""
        new_df = pd.DataFrame(new_data)

        # Binary encoding
        for col in self.binary_cols:
            if col in new_df.columns:
                new_df[col] = new_df[col].map({'yes': 1, 'no': 0})

        # One-hot encoding
        new_df = pd.get_dummies(new_df, columns=['furnishingstatus'], drop_first=True)

        # Align columns with training data
        missing_cols = set(self.feature_names) - set(new_df.columns)
        for col in missing_cols:
            new_df[col] = 0

        # Reorder columns to match training data
        return new_df[self.feature_names]

    def predict(self, new_data):
        """Make predictions on new data"""
        processed_data = self.prepare_new_data(new_data)
        print(f'REG PRE: {self.modelReg.predict(processed_data)}')
        return self.model.predict(processed_data)

    def run_pipeline(self):
        """Execute full pipeline"""
        self.load_data()
        self.preprocess_data()
        self.remove_outliers()
        self.split_data()
        self.train_model()
        self.evaluate_model()
        self.evaluate_model_reg()


# Usage example
if __name__ == "__main__":
    # Configuration
    BINARY_COLS = ['mainroad', 'guestroom', 'basement',
                   'hotwaterheating', 'airconditioning', 'prefarea']
    NUMERIC_COLS = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking', 'price']
    TARGET_COL = 'price'
    DATA_PATH = 'Housing.csv'

    # Initialize predictor
    predictor = HousePricePredictor(DATA_PATH, BINARY_COLS, NUMERIC_COLS, TARGET_COL)

    # Run full pipeline
    predictor.run_pipeline()

    # New data prediction
    new_data = {
        'area': [6000],
        'bedrooms': [4],
        'bathrooms': [3],
        'stories': [2],
        'mainroad': ['yes'],
        'guestroom': ['yes'],
        'basement': ['yes'],
        'hotwaterheating': ['yes'],
        'airconditioning': ['no'],
        'parking': [2],
        'prefarea': ['no'],
        'furnishingstatus': ['semi-furnished']
    }

    prediction = predictor.predict(new_data)
    print(f"\nPredicted Price: {prediction[0]}")