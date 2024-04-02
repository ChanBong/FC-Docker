FROM ubuntu:22.04

ENV HOME /root

RUN apt update \
		&& apt install nginx lsof python3 python3-pip -y \
		&& echo "daemon off;" >> /etc/nginx/nginx.conf 
RUN apt install xfonts-utils --no-install-recommends -y
RUN apt install xfonts-base xfonts-100dpi xfonts-75dpi xfonts-cyrillic --no-install-recommends -y
RUN apt-get install -y gconf-service libasound2 libatk1.0-0 libc6 libcairo2 libcups2 libdbus-1-3 libexpat1 libfontconfig1 libgcc1 libgconf-2-4 libgdk-pixbuf2.0-0 libglib2.0-0 libgtk-3-0 libnspr4 libpango-1.0-0 libpangocairo-1.0-0 libstdc++6 libx11-6 libx11-xcb1 libxcb1 libxcomposite1 libxcursor1 libxdamage1 libxext6 libxfixes3 libxi6 libxrandr2 libxrender1 libxss1 libxtst6 ca-certificates fonts-liberation libappindicator1 libnss3 lsb-release xdg-utils wget libgbm1
RUN pip3 install docker
RUN apt install -y supervisor
ADD html /html

ADD supervisord.conf /etc/supervisor/conf.d/supervisord.conf
#CMD ["/usr/bin/supervisord"]


ADD localhost.conf /etc/nginx/sites-available/default
ADD manager /manager
CMD ls /fcUsers/
CMD nginx & python3 /manager/sessions.py
EXPOSE 80
EXPOSE 1069
