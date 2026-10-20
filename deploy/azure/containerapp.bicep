// Azure Container Apps deployment. Postgres Flexible Server (with the `vector` extension allow-listed)
// and Azure Cache for Redis are provisioned separately; their connection strings are passed in as secure params.
@description('Deployment location')
param location string = resourceGroup().location
@description('Container image, e.g. myacr.azurecr.io/banking-rag:abc123')
param image string
param acrLoginServer string
@secure()
param acrUsername string
@secure()
param acrPassword string
@secure()
param databaseUrl string
@secure()
param redisUrl string
@secure()
param jwtSecret string
@secure()
param anthropicApiKey string
param llmModel string = 'claude-sonnet-4-5'

resource logs 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name: 'banking-rag-logs'
  location: location
  properties: { sku: { name: 'PerGB2018' }, retentionInDays: 30 }
}

resource env 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: 'banking-rag-env'
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logs.properties.customerId
        sharedKey: logs.listKeys().primarySharedKey
      }
    }
  }
}

resource app 'Microsoft.App/containerApps@2024-03-01' = {
  name: 'banking-rag'
  location: location
  properties: {
    managedEnvironmentId: env.id
    configuration: {
      ingress: { external: true, targetPort: 8000, transport: 'auto' }
      registries: [{ server: acrLoginServer, username: acrUsername, passwordSecretRef: 'acr-password' }]
      secrets: [
        { name: 'acr-password', value: acrPassword }
        { name: 'database-url', value: databaseUrl }
        { name: 'redis-url', value: redisUrl }
        { name: 'jwt-secret', value: jwtSecret }
        { name: 'anthropic-api-key', value: anthropicApiKey }
      ]
    }
    template: {
      containers: [
        {
          name: 'api'
          image: image
          resources: { cpu: json('1.0'), memory: '2Gi' }
          env: [
            { name: 'ENV', value: 'prod' }
            { name: 'ALLOW_REGISTRATION', value: 'false' }
            { name: 'LLM_MODEL', value: llmModel }
            { name: 'EMBEDDING_PROVIDER', value: 'fastembed' }
            { name: 'EMBEDDING_MODEL', value: 'BAAI/bge-small-en-v1.5' }
            { name: 'EMBEDDING_DIM', value: '384' }
            { name: 'FASTEMBED_CACHE_PATH', value: '/app/.cache/fastembed' }
            { name: 'DATABASE_URL', secretRef: 'database-url' }
            { name: 'REDIS_URL', secretRef: 'redis-url' }
            { name: 'JWT_SECRET', secretRef: 'jwt-secret' }
            { name: 'ANTHROPIC_API_KEY', secretRef: 'anthropic-api-key' }
          ]
          probes: [
            { type: 'Liveness', httpGet: { path: '/api/v1/health', port: 8000 }, initialDelaySeconds: 60, periodSeconds: 30 }
            { type: 'Readiness', httpGet: { path: '/api/v1/ready', port: 8000 }, initialDelaySeconds: 60, periodSeconds: 15 }
          ]
        }
      ]
      scale: {
        minReplicas: 1
        maxReplicas: 3
        rules: [{ name: 'http', http: { metadata: { concurrentRequests: '40' } } }]
      }
    }
  }
}

output url string = 'https://${app.properties.configuration.ingress.fqdn}'
