from mail import *
import argparse
    
def main():
    local_time = get_time()
    mail_server = get_mail_server()
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', type=str, help='the subject of the email')
    parser.add_argument('--event', type=str, help='the event of the training that will be used in the email body')
    opts = parser.parse_args()
        
    reply = f"{opts.event} started at {local_time}."
    
    for dest in get_mail_addresses():
        send_mail(server = mail_server["server_address"],
                  port = mail_server["server_port"],
                  user = mail_server["user"],
                  password = mail_server["password"],
                  to = dest,
                  subject = opts.subject,
                  body = reply)
                  
if __name__ == '__main__':
    main()
