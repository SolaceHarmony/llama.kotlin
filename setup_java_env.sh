#!/bin/bash

# Script to set up Java environment for llama.kotlin project
# Starts from root directory (/) and sets up environment in /workspace/llama.kotlin

set -e  # Exit on any error

# Check if script is run as root
if [ "$(id -u)" -ne 0 ]; then
    echo "This script must be run as root" >&2
    exit 1
fi

echo "Setting up Java environment for llama.kotlin project..."

# Navigate to project directory
cd /workspace/llama.kotlin
echo "Changed to project directory: $(pwd)"

# Install necessary tools
echo "Installing necessary tools..."
apt-get update
apt-get install -y wget unzip curl gnupg software-properties-common

# Install Java 17 (compatible with Kotlin 2.0.0 and Gradle 8.13)
echo "Installing Java 17..."
add-apt-repository -y ppa:openjdk-r/ppa
apt-get update
apt-get install -y openjdk-17-jdk

# Verify Java installation
java -version
# Set up JAVA_HOME
export JAVA_HOME=$(readlink -f /usr/bin/java | sed "s:/bin/java::")
echo "export JAVA_HOME=$JAVA_HOME" >> /etc/profile.d/java_env.sh
echo "Java installation completed. JAVA_HOME set to $JAVA_HOME"

# Extract Gradle version from wrapper properties
GRADLE_WRAPPER_PROPERTIES="gradle/wrapper/gradle-wrapper.properties"
if [ -f "$GRADLE_WRAPPER_PROPERTIES" ]; then
    GRADLE_URL=$(grep "distributionUrl" "$GRADLE_WRAPPER_PROPERTIES" | cut -d '=' -f2 | sed 's/\\:/:/g')
    GRADLE_VERSION=$(echo "$GRADLE_URL" | grep -o 'gradle-[0-9.]*' | sed 's/gradle-//')
    echo "Detected Gradle version: $GRADLE_VERSION"
else
    echo "Gradle wrapper properties not found. Using default Gradle version."
    GRADLE_VERSION="8.13"
fi

# Install Gradle
echo "Installing Gradle $GRADLE_VERSION..."
wget -q "https://services.gradle.org/distributions/gradle-${GRADLE_VERSION}-bin.zip" -O /tmp/gradle.zip
mkdir -p /opt/gradle
unzip -q /tmp/gradle.zip -d /opt/gradle
mv /opt/gradle/gradle-${GRADLE_VERSION} /opt/gradle/latest
ln -s /opt/gradle/latest/bin/gradle /usr/bin/gradle
rm /tmp/gradle.zip

# Verify Gradle installation
gradle --version
# Set up GRADLE_HOME
export GRADLE_HOME=/opt/gradle/latest
echo "export GRADLE_HOME=$GRADLE_HOME" >> /etc/profile.d/gradle_env.sh
echo "export PATH=\$PATH:\$GRADLE_HOME/bin" >> /etc/profile.d/gradle_env.sh
echo "Gradle installation completed. GRADLE_HOME set to $GRADLE_HOME"

# Run Maven updates/checks if needed
# Note: The project appears to be using Gradle, not Maven, but including this for completeness
echo "Checking for Maven projects..."
if [ -f "pom.xml" ]; then
    echo "Maven project detected. Installing Maven..."
    apt-get install -y maven
    echo "Running Maven updates..."
    mvn clean install -DskipTests
    echo "Maven updates completed."
else
    echo "No Maven project detected. Skipping Maven updates."
fi

# Final verification
echo "Performing final verification..."
echo "Java version:"
java -version
echo "Gradle version:"
gradle --version

echo "Environment setup completed successfully!"
echo "Java and Gradle have been installed and configured for the llama.kotlin project."
