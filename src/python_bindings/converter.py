# Convert a numpy object into bytearray that represents its data.  If the array is 2D and index optional argument
# is not -1, get the data at the given index.
def convert_to_bytearray(np_arr, index=-1):
        if index == -1:
                return bytearray(np_arr)

        return bytearray(np_arr[index])
