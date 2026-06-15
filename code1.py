import matplotlib.pyplot as plt

# Generate numbers and their binary representations interpreted as decimals
x_values = list(range(1, 100))  # Change range as needed
y_values = [int(bin(x)[2:]) for x in x_values]  # Convert to binary and interpret as decimal

# Plot the graph
plt.figure(figsize=(10, 5))
plt.plot(x_values, y_values, marker='o', linestyle='-')

# Labeling
plt.xlabel("Number (x)")
plt.ylabel("Binary representation interpreted as decimal (y)")
plt.title("Number vs Binary Representation Taken as Decimal")
plt.grid(True)

# Show plot
plt.show()
