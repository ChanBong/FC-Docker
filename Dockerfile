FROM ubuntu:22.04

ENV HOME /root

RUN apt update 
RUN apt install software-properties-common --no-install-recommends -y
RUN apt-get install gnupg-agent --no-install-recommends -y
RUN add-apt-repository ppa:freecad-maintainers/freecad-stable -y
RUN apt update
RUN apt install xvfb --no-install-recommends -y
RUN apt install x11vnc --no-install-recommends -y
RUN apt install xdotool --no-install-recommends -y
RUN apt install software-properties-common --no-install-recommends -y
RUN apt install supervisor --no-install-recommends -y
RUN apt install nginx x11-xserver-utils --no-install-recommends -y
RUN apt install xterm --no-install-recommends -y
RUN apt install freecad --no-install-recommends -y
RUN apt install jwm --no-install-recommends -y
RUN apt install xfonts-utils --no-install-recommends -y
RUN apt install xfonts-base xfonts-100dpi xfonts-75dpi xfonts-cyrillic --no-install-recommends -y
RUN apt-get clean autoclean 
RUN apt-get autoremove --yes 
RUN apt-get install -y gconf-service libasound2 libatk1.0-0 libc6 libcairo2 libcups2 libdbus-1-3 libexpat1 libfontconfig1 libgcc1 libgconf-2-4 libgdk-pixbuf2.0-0 libglib2.0-0 libgtk-3-0 libnspr4 libpango-1.0-0 libpangocairo-1.0-0 libstdc++6 libx11-6 libx11-xcb1 libxcb1 libxcomposite1 libxcursor1 libxdamage1 libxext6 libxfixes3 libxi6 libxrandr2 libxrender1 libxss1 libxtst6 ca-certificates fonts-liberation libappindicator1 libnss3 lsb-release xdg-utils wget libgbm1
RUN rm -rf /var/lib/{apt,dpkg,cache,log}/


RUN apt install \
		firefox \
		mousepad \
		thunar \
		nano \
		--no-install-recommends -y 

# ERRORS OUT HERE
RUN apt install git -y 
# RUN apt install python-numpy python-pyside -y

ADD external_dependencies/external/Mod /root/.FreeCAD/Mod 

ADD system.jwmrc /etc/jwm/system.jwmrc

EXPOSE 80
EXPOSE 1069

WORKDIR /root/

ENV DISPLAY :0

ADD external_dependencies/external/Droopy /fileServer
ADD external_dependencies/external/noVNC /novnc
ADD external_dependencies/external/icons /novnc/icons

ADD supervisord.conf /etc/supervisor/conf.d/supervisord.conf
ADD start.sh /start.sh

ADD localhost.conf /etc/nginx/sites-available/default


ADD resolution.py /resolution.py
CMD ["/usr/bin/supervisord"]


ADD server.py /server.py

ADD user.cfg /root/.FreeCAD/user.cfg

ADD settings.json /novnc/settings.json
ADD mbd_test.py /mbd_test.py

RUN ln /root/.FreeCAD/user.cfg /root/user.cfg
ADD index.html /novnc/index.html
ADD FC-APP.js /novnc/FC-APP.js
