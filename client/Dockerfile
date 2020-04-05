FROM node:12.13.1

## set working directory
WORKDIR /app/client

## add `/usr/src/app/node_modules/.bin` to $PATH
ENV PATH /app/client/node_modules/.bin:$PATH

## install and cache app dependencies
## create user "node" and give permissions
COPY package.json /app/client/package.json
RUN npm install
RUN npm install react-scripts@0.9.5 -g

# RUN chown -R node:node . && chmod -R 755 .
# USER node
# RUN npm install --silent
# RUN npm cache clean --force

## start app
## see package.json for npm command
CMD ["npm", "start"]