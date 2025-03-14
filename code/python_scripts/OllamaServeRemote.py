from secrets_config.secret_variables import SERVER_IP
from secrets_config.secret_variables import USERNAME
from secrets_config.secret_variables import PASSWORD

import paramiko

# Creates the SSH Client
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

try:
    print(f"Trying connection to {SERVER_IP}")

    # Create SSH client
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    # TODO: Connection does not work yet, haven't been able to figure out why
    print(f"Connecting to {SERVER_IP}...")
    ssh.connect(
        SERVER_IP, 
        username=USERNAME, 
        password=PASSWORD, 
    )

    # Setting OLLAMA_HOST=0.0.0.0 makes the Ollama accessible from computers in the same network
    # Calling ollama serve starts the ollama instance
    # > /dev/null 2>&1 & discards unneccessary output
    # TODO: So far only tested manually (in a form that is a bit different) since python connection doesn't work yet
    print("Connected! Running command...")
    command = "export OLLAMA_HOST=0.0.0.0 && nohup ollama serve > /dev/null 2>&1 &"
    stdin, stdout, stderr = ssh.exec_command(command)
    
    print("Ollama started!")
    ssh.close()

except Exception as e:
    print(f"Error: {e}")
