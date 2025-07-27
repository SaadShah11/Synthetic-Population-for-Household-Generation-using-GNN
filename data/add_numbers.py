def sum_comma_separated_numbers(input_str):
    # Split the string by commas, convert each to int, and sum them
    numbers = map(int, input_str.split(','))
    return sum(numbers)

# Example input
input_string = "36,68,137,19,68,49,4,21,45,84,15,98,229,11,63,144,131,57,172,472,28,114,93,66,106,107,329,35,176,36,49,111,41,184,43,25,54,126,7,84,58,2,25,44,76,3,106,274,7,55,169,220,5,162,510,14,145,136,142,10,83,324,36,316,58,83,12,18,143,46"

# Calculate sum
total = sum_comma_separated_numbers(input_string)
print("Total sum:", total)
