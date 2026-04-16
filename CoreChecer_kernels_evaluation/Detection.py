
def main():
    # job_id = os.getenv('SLURM_JOB_ID')
    Detection_Result_Log = f"./control/0/SM_checking_results.txt"
    # Banned_SMID_Log = f"./control/0/banned_smid.txt"
    # logFP = f"./control_{job_id}/0/output.log"
    
    fault_detection_res = []

    with open(Detection_Result_Log, 'r') as file:
        # faulty_step = int(file.readline())
        for line in file:
            # print(line)
            str_list = line.strip().split()
            fault_detection_res = [int(e) for e in str_list]  

    # print(fault_detection_res)
    
    all_zero = all(x == 0 for x in fault_detection_res)

    if not all_zero:
        max_val = max(fault_detection_res)
        if fault_detection_res.count(max_val) == 1:
            faulty_smid = fault_detection_res.index(max_val)
            print(f"CoreChecker Detected Faulty SMID: {faulty_smid}")
            
            # with open(Banned_SMID_Log, 'w') as file:
            #     file.write(str(faulty_smid))
            
            # with open(logFP, 'a') as file:
            #     file.write(f"{faulty_smid}\n")
    
            # with open(Detection_Result_Log, 'w') as file:
            #     file.truncate(0)

            return True
        else:
            # with open(Detection_Result_Log, 'w') as file:
            #     file.truncate(0)
            print("No Faulty SM Detected.")
            return False 
    else:
        # with open(Detection_Result_Log, 'w') as file:
        #     file.truncate(0)
        print("No Faulty SM Detected.")
        return False


if __name__ == "__main__":
    main()