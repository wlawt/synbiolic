FROM node:12.13.1

## set working directory
WORKDIR /app/server

## add `/usr/src/app/node_modules/.bin` to $PATH
ENV PATH /app/server/node_modules/.bin:$PATH

## install and cache app dependencies
## create user "node" and give permissions
COPY package.json /app/server/package.json
RUN npm install --global nodemon

RUN npm install

# RUN chown -R node:node . && chmod -R 755 .
# USER node
# RUN npm install --silent
# RUN npm cache clean --force

## start app
## see package.json for npm command
CMD ["nodemon","server.js"]