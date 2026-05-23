"""
Tests for DuckJanitor class.
"""

import pytest
import pandas as pd
import numpy as np
from pyduck_janitor import DuckJanitor


class TestDuckJanitor:
    """Tests for the DuckJanitor class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample DataFrame for testing."""
        return pd.DataFrame({
            'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
            'Age': [25, 30, 35, 40],
            'City': ['NYC', 'LA', 'Chicago', 'Houston'],
            'Salary': [50000, 60000, 70000, 80000],
        })
    
    @pytest.fixture
    def data_with_nulls(self):
        """Create sample DataFrame with missing values."""
        return pd.DataFrame({
            'A': [1, 2, None, 4, 5],
            'B': [None, 2, 3, 4, 5],
            'C': [1, 2, 3, None, 5],
        })
    
    def test_from_pandas(self, sample_data):
        """Test creating DuckJanitor from pandas DataFrame."""
        dj = DuckJanitor.from_pandas(sample_data)
        
        assert isinstance(dj, DuckJanitor)
        assert len(dj.head()) == len(sample_data)
    
    def test_collect(self, sample_data):
        """Test collecting results back to pandas."""
        dj = DuckJanitor.from_pandas(sample_data)
        result = dj.collect()
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape == sample_data.shape
    
    def test_head(self, sample_data):
        """Test head method."""
        dj = DuckJanitor.from_pandas(sample_data)
        result = dj.head(3)
        
        assert len(result) == 3
    
    def test_clean_names(self, sample_data):
        """Test cleaning column names."""
        # Create data with messy column names
        messy = sample_data.copy()
        messy.columns = ['  First Name  ', 'Age-Years', 'CITY!', 'Annual Salary']
        
        dj = DuckJanitor.from_pandas(messy)
        cleaned = dj.clean_names()
        result = cleaned.collect()
        
        expected_cols = ['first_name', 'age_years', 'city', 'annual_salary']
        assert list(result.columns) == expected_cols
    
    def test_remove_columns(self, sample_data):
        """Test removing columns."""
        dj = DuckJanitor.from_pandas(sample_data)
        result = dj.remove_columns(['City', 'Salary']).collect()
        
        assert 'City' not in result.columns
        assert 'Salary' not in result.columns
        assert 'Name' in result.columns
        assert 'Age' in result.columns
    
    def test_rename_column(self, sample_data):
        """Test renaming a column."""
        dj = DuckJanitor.from_pandas(sample_data)
        result = dj.rename_column('Name', 'Full_Name').collect()
        
        assert 'Full_Name' in result.columns
        assert 'Name' not in result.columns
    
    def test_add_column_scalar(self, sample_data):
        """Test adding a column with a scalar value."""
        dj = DuckJanitor.from_pandas(sample_data)
        result = dj.add_column('Constant', 42).collect()
        
        assert 'Constant' in result.columns
        assert (result['Constant'] == 42).all()
    
    def test_add_column_expression(self, sample_data):
        """Test adding a column with a SQL expression."""
        dj = DuckJanitor.from_pandas(sample_data)
        result = dj.add_column('Double_Age', 'Age * 2').collect()
        
        assert 'Double_Age' in result.columns
        assert (result['Double_Age'] == sample_data['Age'] * 2).all()
    
    def test_dropna(self, data_with_nulls):
        """Test dropping rows with missing values."""
        dj = DuckJanitor.from_pandas(data_with_nulls)
        result = dj.dropna().collect()
        
        assert result.isnull().sum().sum() == 0
    
    def test_dropna_subset(self, data_with_nulls):
        """Test dropping rows with missing values in subset."""
        dj = DuckJanitor.from_pandas(data_with_nulls)
        result = dj.dropna(subset=['A']).collect()
        
        assert result['A'].isnull().sum() == 0
    
    def test_remove_empty(self, data_with_nulls):
        """Test removing empty rows and columns."""
        dj = DuckJanitor.from_pandas(data_with_nulls)
        result = dj.remove_empty().collect()
        
        assert len(result) > 0
    
    def test_filter_column_callable(self, sample_data):
        """Test filtering with a callable."""
        dj = DuckJanitor.from_pandas(sample_data)
        result = dj.filter_column('Age', lambda x: x > 30).collect()
        
        assert (result['Age'] > 30).all()
    
    def test_filter_column_sql(self, sample_data):
        """Test filtering with SQL string."""
        dj = DuckJanitor.from_pandas(sample_data)
        result = dj.filter_column('Age', 'Age > 30').collect()
        
        assert (result['Age'] > 30).all()
    
    def test_coalesce(self, data_with_nulls):
        """Test coalescing columns."""
        dj = DuckJanitor.from_pandas(data_with_nulls)
        result = dj.coalesce(['A', 'B', 'C'], 'coalesced').collect()
        
        assert 'coalesced' in result.columns
        assert result['coalesced'].isnull().sum() == 0
    
    def test_encode_categorical(self, sample_data):
        """Test encoding categorical column."""
        dj = DuckJanitor.from_pandas(sample_data)
        result = dj.encode_categorical('City').collect()
        
        assert 'City_cat' in result.columns
    
    def test_get_dummies(self, sample_data):
        """Test one-hot encoding."""
        dj = DuckJanitor.from_pandas(sample_data)
        result = dj.get_dummies('City', prefix='city').collect()
        
        # Should have dummy columns for each city
        assert any(col.startswith('city_') for col in result.columns)
    
    def test_sql(self, sample_data):
        """Test custom SQL query."""
        dj = DuckJanitor.from_pandas(sample_data)
        result = dj.sql("SELECT * FROM self WHERE Age > 30").collect()
        
        assert (result['Age'] > 30).all()
    
    def test_method_chaining(self, sample_data):
        """Test method chaining."""
        dj = DuckJanitor.from_pandas(sample_data)
        result = (
            dj
            .clean_names()
            .remove_columns(['city'])
            .rename_column('name', 'full_name')
            .add_column('double_salary', 'salary * 2')
            .filter_column('age', 'age > 25')
            .collect()
        )
        
        assert 'full_name' in result.columns
        assert 'city' not in result.columns
        assert 'double_salary' in result.columns
        assert (result['age'] > 25).all()
    
    def test_explain(self, sample_data):
        """Test query plan explanation."""
        dj = DuckJanitor.from_pandas(sample_data)
        explanation = dj.explain()
        
        assert isinstance(explanation, str)
        assert len(explanation) > 0
    
    def test_repr(self, sample_data):
        """Test string representation."""
        dj = DuckJanitor.from_pandas(sample_data)
        repr_str = repr(dj)
        
        assert 'DuckJanitor' in repr_str
        assert 'lazy' in repr_str.lower()
