import joblib
import pandas as pd

# Define file paths for model and preprocessor
MODEL_PATH = "backpack_price_model.pkl"
PREPROCESSOR_PATH = "preprocessor.pkl"

# Load the trained model and preprocessor
try:
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
except FileNotFoundError:
    print("Error: Model or preprocessor file not found. Make sure both are available.")
    exit()

def get_user_input():
    """ Function to get backpack details from the user. """
    print("\nEnter backpack details:")
    
    brand = input("Brand: ")
    material = input("Material: ")
    size = input("Size (Small/Medium/Large): ")
    compartments = int(input("Number of Compartments: "))
    laptop_compartment = input("Laptop Compartment (Yes/No): ")
    waterproof = input("Waterproof (Yes/No): ")
    style = input("Style (Casual/Travel/Sports/etc.): ")
    color = input("Color: ")
    weight_capacity = float(input("Weight Capacity (kg): "))

    # Create a DataFrame with the input values
    user_data = pd.DataFrame([{
        "Brand": brand,
        "Material": material,
        "Size": size,
        "Compartments": compartments,
        "Laptop Compartment": laptop_compartment,
        "Waterproof": waterproof,
        "Style": style,
        "Color": color,
        "Weight Capacity (kg)": weight_capacity
    }])

    return user_data

def predict_price():
    """ Function to predict price based on user input. """
    user_data = get_user_input()

    # Preprocess the user input
    user_data_preprocessed = preprocessor.transform(user_data)

    # Predict the price
    predicted_price = model.predict(user_data_preprocessed)[0]
    
    print(f"\nPredicted Price: {predicted_price:.2f} USD")

def main():
    """ Main function to run the CLI tool. """
    print("===== Backpack Price Prediction Tool =====")
    
    while True:
        predict_price()
        
        # Ask user if they want to predict another price
        another = input("\nDo you want to predict another price? (yes/no): ").strip().lower()
        if another != "yes":
            print("Thank you for using the Backpack Price Prediction tool!")
            break

if __name__ == "__main__":
    main()
