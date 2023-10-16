FROM node:alpine

WORKDIR /app

COPY front-end/ .

RUN npm install

EXPOSE 8505

ENTRYPOINT [ "npm", "run", "dev" ]
