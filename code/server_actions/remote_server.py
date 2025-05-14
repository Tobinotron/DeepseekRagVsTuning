from secrets_config.secret_variables import SERVER_IP
from secrets_config.secret_variables import USERNAME
from secrets_config.secret_variables import PASSWORD

import paramiko
import sys
import os

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

def execute_command_on_remote(command):
    stdin, stdout, stderr = ssh.exec_command(command)
    return stdin, stdout, stderr

def push_script_to_remote(local_script_path, remote_folder="~/py_scripts"):
    """
    Transfers a local script to a specified folder on the remote server.

    Parameters:
    - local_script_path (str): Path to the script on the local machine.
    - remote_folder (str): Path to the target folder on the remote server.
    """
    if not ssh:
        print("No SSH connection established.")
        return

    try:
        # Normalize paths
        local_script_path = os.path.abspath(local_script_path)  # Ensure full Windows path
        script_filename = os.path.basename(local_script_path)  # Extract just the filename

        # Expand ~ on remote server
        stdin, stdout, stderr = ssh.exec_command(f"echo {remote_folder}")
        remote_folder = stdout.read().decode().strip()  # Resolve `~` to full path

        remote_script_path = f"{remote_folder}/{script_filename}"

        print(f"Uploading {local_script_path} to {remote_script_path}...")

        # Open SFTP session
        sftp = ssh.open_sftp()

        # Upload the file
        sftp.put(local_script_path.replace("\\", "/"), remote_script_path)

        # Ensure the script has execute permissions
        ssh.exec_command(f"chmod +x {remote_script_path}")

        print(f"Script uploaded successfully to {remote_script_path}")

        # Close SFTP session
        sftp.close()

        return remote_script_path

    except Exception as e:
        print(f"Failed to upload script: {e}")

def run_script_on_remote(script_to_run, interpreter="Ollama"):
    """
    Runs a given Python script on the remote server using the specified interpreter and streams the output.

    Parameters:
    - script_to_run (str): The absolute path to the script on the remote machine.
    - interpreter (str): The name of the Python interpreter (default: "Ollama").
    """
    if not ssh:
        print("No SSH connection established.")
        return

    # Construct the command to run the script with the specified interpreter
    command = f"bash -l -c 'source ~/anaconda3/bin/activate {interpreter} && python {script_to_run}'"

    print(f"Running script '{script_to_run}' on remote server with interpreter '{interpreter}'")

    try:
        stdin, stdout, stderr = ssh.exec_command(command)

        # Stream stdout and stderr in real time
        while True:
            output_line = stdout.readline()
            error_line = stderr.readline()

            if not output_line and not error_line:
                break  # Exit when there's no more output

            if output_line:
                print(output_line, end="")  # Print without extra newlines
                sys.stdout.flush()  # Ensure immediate output

            if error_line:
                print(f"ERROR: {error_line}", end="", file=sys.stderr)
                sys.stderr.flush()

    except Exception as e:
        print(f"Failed to execute script: {e}")

if __name__ == "__main__":
    establish_connection()
    start_ollama()
    terminate_connection()