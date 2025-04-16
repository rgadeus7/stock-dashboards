import os
import pandas as pd
from bias_analyzer import BiasAnalyzer
from file_manager import FileManager

def test_bias_analyzer():
    # Initialize FileManager and BiasAnalyzer
    file_manager = FileManager()
    analyzer = BiasAnalyzer(file_manager)
    
    # Create test data with different scenarios
    test_data = {
        'Date': pd.date_range(start='2024-01-01', periods=10, freq='D'),
        'Open': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 104.0, 103.0, 102.0, 101.0],
        'High': [102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 106.0, 105.0, 104.0, 103.0],
        'Low': [99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 103.0, 102.0, 101.0, 100.0],
        'Close': [101.0, 102.0, 103.0, 104.0, 105.0, 104.0, 103.0, 102.0, 101.0, 100.0],
        'Volume': [1000, 1100, 1200, 1300, 1400, 1300, 1200, 1100, 1000, 900]
    }
    
    # Create test DataFrame
    test_df = pd.DataFrame(test_data)
    
    # Save test data to CSV
    test_file_path = os.path.join(file_manager.data_dir, 'test_data.csv')
    test_df.to_csv(test_file_path, index=False)
    
    try:
        # Run the analyzer
        print("\n=== Testing BiasAnalyzer ===")
        print("1. Loading and analyzing test data...")
        results = analyzer.analyze_and_save(test_file_path)
        
        # Print results
        print("\n2. Analysis Results:")
        print("\nBias Hit Summary:")
        for item in results['bias_hit_summary']:
            print(f"Bias: {item['Bias']}, Count: {item['Count']}, Hit Count: {item['Hit_Count']}, Hit Rate: {item['Hit_Rate_Percent']}%")
        
        print("\nClose Hit Summary:")
        for item in results['close_hit_summary']:
            print(f"Type: {item['Close_Hit_Type']}, Count: {item['Count']}, Percent: {item['Percent']}%")
        
        print("\nBias Reason Summary:")
        for item in results['bias_reason_summary']:
            print(f"Reason: {item['Bias_Reason']}, Count: {item['Count']}, Percentage: {item['Percentage']}%")
        
        # Verify the output file was created
        output_file = os.path.join(file_manager.analysis_dir, f"bias_analysis_{pd.Timestamp.now().strftime('%Y-%m-%d')}.json")
        if os.path.exists(output_file):
            print(f"\n3. Success! Output file created at: {output_file}")
        else:
            print("\n3. Error: Output file was not created")
        
    except Exception as e:
        print(f"\nError during testing: {str(e)}")
        raise  # Re-raise the exception to see the full traceback
    
    finally:
        # Clean up test files
        if os.path.exists(test_file_path):
            os.remove(test_file_path)
        print("\nTest files cleaned up")

if __name__ == "__main__":
    test_bias_analyzer() 