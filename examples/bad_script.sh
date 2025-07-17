#!/bin/bash
# Example shell script with intentional bugs and issues

# Missing 'set -e' for error handling

# Hardcoded credentials
PASSWORD="supersecret123"
API_TOKEN="abc-123-secret-token-456"

# TODO: Move credentials to environment variables

# Unquoted variables (word splitting issues)
function process_file() {
    local filename=$1
    
    # Unquoted variable usage
    if [ -f $filename ]; then
        cat $filename | grep something
    fi
    
    # Another unquoted variable
    cp $filename $filename.backup
}

# Function with potential issues
function deploy_application() {
    local app_name=$1
    local environment=$2
    
    # Unquoted variables in command
    ssh user@server "cd /opt/$app_name && ./deploy.sh $environment"
    
    # Command that might fail but no error checking
    curl -X POST "https://api.example.com/deploy" -H "Authorization: Bearer $API_TOKEN"
}

# Line that is way too long and should be flagged by the line length checker - this demonstrates poor shell script formatting
very_long_command_that_exceeds_reasonable_line_length="This is a very long assignment that makes the line too long for good readability"

# Main execution without proper error handling
echo "Starting deployment script"
echo "Using password: $PASSWORD"  # Logging sensitive information

process_file config.txt
deploy_application myapp production

echo "Deployment completed"

# FIXME: Add proper error handling
# TODO: Implement logging
# HACK: Temporary solution for permissions