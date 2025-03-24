document.addEventListener("DOMContentLoaded", function() {
    console.log("Deconstructionist Cookbook – Savor Your Story!");

    const interactiveElements = document.querySelectorAll('.interactive-button, .recipe-card, .landing-card');
    interactiveElements.forEach(element => {
        element.addEventListener('mouseover', function() {
            this.style.transform = 'scale(1.05)';
        });
        element.addEventListener('mouseout', function() {
            this.style.transform = 'scale(1)';
        });
    });

    const flavorRadarBtn = document.getElementById('flavorRadarBtn');
    if (flavorRadarBtn) {
        flavorRadarBtn.addEventListener('click', function() {
            const radarSection = document.getElementById('graph-radar');
            if (radarSection) {
                radarSection.scrollIntoView({ behavior: 'smooth' });
            }
        });
    }

    const likeForms = document.querySelectorAll('.like-form');
    likeForms.forEach(form => {
        form.addEventListener('submit', function(e) {
            e.preventDefault();
            const formData = new FormData(form);
            fetch(form.action, {
                method: 'POST',
                body: formData
            })
            .then(response => {
                if (response.redirected) {
                    const button = form.querySelector('.like-button');
                    button.innerHTML = '<i class="fa fa-heart" aria-hidden="true"></i>';
                    button.classList.add('liked');
                    button.disabled = true;
                    window.location.reload(); // Refresh to update liked recipes tab
                }
            })
            .catch(error => console.error('Error:', error));
        });
    });
});