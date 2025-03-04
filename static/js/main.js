// main.js

document.addEventListener("DOMContentLoaded", function() {
    console.log("Deconstructionist Cookbook loaded – unveiling your culinary identity through ingredients!");

    // Lightweight hover effect for buttons and cards
    const interactiveElements = document.querySelectorAll('.interactive-button, .recipe-card');
    interactiveElements.forEach(element => {
        element.addEventListener('mouseover', function() {
            this.style.transform = 'scale(1.02)';
            this.style.transition = 'transform 0.2s ease';
        });
        element.addEventListener('mouseout', function() {
            this.style.transform = 'scale(1)';
        });
    });

    // Handle Flavor Profile Radar button (switch to Visuals tab)
    const flavorRadarBtn = document.getElementById('flavorRadarBtn');
    flavorRadarBtn.addEventListener('click', function() {
        document.getElementById('visuals-tab').click();  // Switch to Visuals tab
        this.style.transform = 'scale(1.05)';
        setTimeout(() => this.style.transform = 'scale(1)', 200);
    });
});