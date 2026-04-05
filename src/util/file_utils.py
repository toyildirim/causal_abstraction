import os
import pickle

import pandas as pd


class FileUtil:
    @staticmethod
    def generate_target_path(source_path, target_dir, extension):
        """
        Generates a .pkl path in the target directory using the source filename.
        Example: '/path/to/data.rda' -> '/target/dir/data.pkl'
        """
        # Extract the filename without extension (e.g., 'experiment_results')
        base_name = os.path.splitext(os.path.basename(source_path))[0]

        # Construct the full path using the provided DATA_PATH
        return os.path.join(target_dir, f"{base_name}"+extension)

    @staticmethod
    def save_as_pkl(data, original_path, output_path):
        """
        Saves the converted data into a binary .pkl file.
        """
        # Default to same folder with .pkl extension
        output_path = FileUtil.generate_target_path(original_path, output_path, ".pkl")
        # base = os.path.splitext(original_path)[0]
        # output_path = f"{base}_from_r.pkl"

        try:
            with open(output_path, 'wb') as f:
                pickle.dump(data, f)
            return output_path
        except Exception as e:
            raise Exception(f"Failed to write Pickle file: {e}")

    @staticmethod
    def get_summary(data):
        """Returns a string summary of the objects found in the RDA."""
        summary_lines = []
        for name, df in data.items():
            if isinstance(df, pd.DataFrame):
                summary_lines.append(f"Object: {name} | Shape: {df.shape}")
        return "\n".join(summary_lines)