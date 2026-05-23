"""
Tests for cleaning operations.
"""

import pytest
import pandas as pd
import numpy as np
import duckdb
from pyduck_janitor.cleaning_ops import (
    clean_names,
    remove_columns,
    add_column,
    rename_column,
    dropna,
    remove_empty,
    filter_column,
    coalesce,
)


class TestCleaningOps:
    """Tests for cleaning operation functions."""
    
    @pytest.fixture
    def conn(self):
        """Create DuckDB connection."""
        return duckdb.connect()
    
    @pytest.fixture
    def sample_relation(self, conn):
        """Create sample DuckDB relation."""
        df = pd.DataFrame({
            'Name': ['Alice', 'Bob', 'Charlie'],
            'Age': [25, 30, 35],
            'City': ['NYC', 'LA', 'Chicago'],
        })
        return conn.from_df(df)
    
    def test_clean_names(self, conn):
        """Test cleaning column names."""
        df = pd.DataFrame({
            '  First Name  ': [1, 2, 3],
            'Last-Name': [4, 5, 6],
            'City!@#': [7, 8, 9],
        })
        relation = conn.from_df(df)
        
        result = clean_names(relation, conn=conn)
        result_df = result.df()
        
        expected_cols = ['first_name', 'last_name', 'city']
        assert list(result_df.columns) == expected_cols
    
    def test_remove_columns(self, conn):
        """Test removing columns."""
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6],
            'C': [7, 8, 9],
        })
        relation = conn.from_df(df)
        
        result = remove_columns(relation, ['B', 'C'], conn=conn)
        result_df = result.df()
        
        assert list(result_df.columns) == ['A']
    
    def test_add_column_scalar(self, conn, sample_relation):
        """Test adding column with scalar value."""
        result = add_column(sample_relation, 'Constant', 42, conn=conn)
        result_df = result.df()
        
        assert 'Constant' in result_df.columns
        assert (result_df['Constant'] == 42).all()
    
    def test_add_column_expression(self, conn, sample_relation):
        """Test adding column with SQL expression."""
        result = add_column(sample_relation, 'Double_Age', 'Age * 2', conn=conn)
        result_df = result.df()
        
        assert 'Double_Age' in result_df.columns
        assert (result_df['Double_Age'] == sample_relation.df()['Age'] * 2).all()
    
    def test_rename_column(self, conn, sample_relation):
        """Test renaming column."""
        result = rename_column(sample_relation, 'Name', 'Full_Name', conn=conn)
        result_df = result.df()
        
        assert 'Full_Name' in result_df.columns
        assert 'Name' not in result_df.columns
    
    def test_dropna(self, conn):
        """Test dropping missing values."""
        df = pd.DataFrame({
            'A': [1, 2, None, 4],
            'B': [None, 2, 3, 4],
        })
        relation = conn.from_df(df)
        
        result = dropna(relation, conn=conn)
        result_df = result.df()
        
        assert result_df.isnull().sum().sum() == 0
    
    def test_dropna_subset(self, conn):
        """Test dropping missing values in subset."""
        df = pd.DataFrame({
            'A': [1, 2, None, 4],
            'B': [None, 2, 3, 4],
        })
        relation = conn.from_df(df)
        
        result = dropna(relation, subset=['A'], conn=conn)
        result_df = result.df()
        
        assert result_df['A'].isnull().sum() == 0
    
    def test_coalesce(self, conn):
        """Test coalescing columns."""
        df = pd.DataFrame({
            'A': [1, None, 3],
            'B': [None, 2, 3],
            'C': [1, 2, 3],
        })
        relation = conn.from_df(df)
        
        result = coalesce(relation, ['A', 'B', 'C'], 'coalesced', conn=conn)
        result_df = result.df()
        
        assert 'coalesced' in result_df.columns
        assert result_df['coalesced'].isnull().sum() == 0
    
    def test_filter_column_sql(self, conn, sample_relation):
        """Test filtering with SQL."""
        result = filter_column(sample_relation, 'Age', 'Age > 25', conn=conn)
        result_df = result.df()
        
        assert (result_df['Age'] > 25).all()
    
    def test_filter_column_callable(self, conn, sample_relation):
        """Test filtering with callable."""
        result = filter_column(sample_relation, 'Age', lambda x: x > 25)
        result_df = result.df()
        
        assert (result_df['Age'] > 25).all()
