import psutil # To monitor RAM usage


def CheckMemoryUsage():
    ramUsage = psutil.virtual_memory()[2]
    print(f"***RAM memory % used: {psutil.virtual_memory()[2]}***")
    if ramUsage >= 85 :
        print("***RAM usage exceeded 85%... Exiting program***")
        exit()