import time
from prettytable import PrettyTable

class Timer:
    def __init__(self, start_msg = "", end_msg = ""):
    
        self.start_msg = start_msg
        self.end_msg = end_msg
        
    def __enter__(self):
        if self.start_msg != "":
            print(self.start_msg)
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time.time() - self.start_time
        print(self.end_msg, f"{elapsed_time:.3f} sec")


def count_parameters(model, print_table = False):
    
    total_params = 0
    
    if(print_table):
        table = PrettyTable(["Modules", "Parameters"]) 
    
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        
        if(print_table):
            table.add_row([name, params])
            
        total_params += params
        
    if(print_table):
        print(table)
        
    if total_params/1e9 > 1:
        print(f"Total Trainable Params: {total_params/1e9} B")
    else:
        print(f"Total Trainable Params: {total_params/1e6} M")
        
    return total_params