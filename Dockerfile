FROM node:20-alpine

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci --omit=dev

COPY src ./src
COPY worker ./worker

EXPOSE 3000
CMD ["npm", "start"]