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

    # Establish Connection to the server
    print(f"Connecting to {SERVER_IP}...")
    ssh.connect(
        SERVER_IP, 
        username=USERNAME, 
        password=PASSWORD, 
    )

    # nohup (should) make paramiko not wait until the process is finished - not working yet
    # Setting OLLAMA_HOST=0.0.0.0 makes the Ollama accessible from computers in the same network
    # Calling ollama serve starts the ollama instance
    # > /dev/null 2>&1 & discards unneccessary output
    # TODO: Make sure the paramiko process finishes
    print("Connected! Running command...")
    command = "nohup bash -l -c 'export OLLAMA_HOST=0.0.0.0 && /home/tobias/ollama/bin/ollama serve > /dev/null 2>&1 &' &"
 
    stdin, stdout, stderr = ssh.exec_command(command)
    print(stderr.read().decode('utf-8'))
    
    # Remark: This does currently never print because the command from above never finishes
    # Ollama does get started though
    print("Ollama started!")
    ssh.close()

except Exception as e:
    print(f"Error: {e}")
