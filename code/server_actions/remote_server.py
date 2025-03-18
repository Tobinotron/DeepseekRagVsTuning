from secrets_config.secret_variables import SERVER_IP
from secrets_config.secret_variables import USERNAME
from secrets_config.secret_variables import PASSWORD

import paramiko

# Global variable to hold SSH Connection
ssh : paramiko.SSHClient = None

def establish_connection():
    """
    Establishes an SSH connection to the remote server using the credentials 
    from the secrets_config module. If successful, stores the connection in the 
    global `ssh` variable for further use.
    """
    global ssh

    try:
        print(f"Trying connection to {SERVER_IP}")

        # Create SSH client
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # Establish Connection to the server
        print(f"Connecting to {SERVER_IP}...")
        ssh.connect(
            SERVER_IP, 
            username=USERNAME, 
            password=PASSWORD, 
        )
        print("Connected to Remote Server")
    
    except Exception as e:
        print(f"Error: {e}")
        # To prevent follow up errors due to error in connection
        ssh = None

def terminate_connection():
    """
    Closes the existing SSH connection, if it exists.
    """
    global ssh

    if ssh:
        ssh.close()
        print("SSH connection closed.")
    else:
        print("No active SSH connection to close.")

def check_ollama_status():
    """
    Checks whether or not the 'ollama serve' process is running on the remote server.
    """
    command = "pgrep -f 'ollama serve'"
    stdin, stdout, stderr = ssh.exec_command(command)

    if stdout.read().decode().strip():
        print("Ollama is running")
    else:
         print("Ollama is not running")

def start_ollama():
    """
    Starts the 'ollama serve' process on the remote server in the background.
    """
    if not ssh:
        print("No SSH connection established.")
        return
    
    print("Running command: Ollama Serve")
    command = "nohup bash -l -c 'export OLLAMA_HOST=0.0.0.0 && /home/tobias/ollama/bin/ollama serve > /dev/null 2>&1 &' & disown"
    
    stdin, stdout, stderr = ssh.exec_command(command)
        
    print("Ollama started!")

def stop_ollama():
     """
    Terminates the 'ollama serve' process.
    """
     command = "pkill -f 'ollama serve'"
     stdin, stdout, stderr = ssh.exec_command(command)
     
     print("Ollama stopped!")

if __name__ == "__main__":
    establish_connection()
    start_ollama()
    terminate_connection()