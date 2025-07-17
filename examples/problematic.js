// Example JavaScript file with intentional bugs and issues

// Using var instead of let/const
var userName = "admin";
var secretKey = "hardcoded-api-key-123456";

// TODO: Fix security issues

function authenticateUser(user, pass) {
    // Using == instead of ===
    if (user == "admin" && pass == "password") {
        console.log("User authenticated successfully");  // Debug statement
        return true;
    }
    
    // Potential XSS vulnerability
    document.getElementById("welcome").innerHTML = "Welcome " + user + "!";
    
    return false;
}

// Security risk: eval usage
function executeCode(code) {
    var result = eval(code);  // Dangerous!
    return result;
}

// Using var in loop (function scope issues)
function processItems() {
    var items = ["a", "b", "c"];
    
    for (var i = 0; i < items.length; i++) {
        setTimeout(function() {
            console.log("Processing item: " + items[i]);  // Will log undefined
        }, 100);
    }
}

// Line that is way too long and should be flagged by the line length checker - this demonstrates poor code formatting and readability issues
var veryLongVariableNameThatMakesThisLineExceedReasonableLength = "This is a very long string that contributes to making this line way too long for good readability";

// Equality comparison issues
function compareValues(a, b) {
    if (a == b) {  // Should use ===
        return true;
    }
    
    if (a != null) {  // Should use !==
        return false;
    }
    
    return a == undefined;  // Should use ===
}

// More debug statements
console.log("Script loaded");
console.log("Secret key: " + secretKey);

// FIXME: Improve error handling
// TODO: Add input validation
// HACK: Quick workaround for IE compatibility